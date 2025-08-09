import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import tempfile
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("gem_api")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
        max_tokens=2048,
        timeout=60
    )
except Exception as e:
    logger.error(f"خطأ في إعداد النموذج: {e}")
    raise

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./embeddings_cache"
    )
except Exception as e:
    logger.warning(f"فشل في تحميل BAAI/bge-m3، سيتم استخدام النموذج الافتراضي: {e}")
    embeddings = HuggingFaceEmbeddings()

PROMPT_TEMPLATE = """
if the question is in english anwser in english 
اذا كان السؤال باللغة العربية اجب العربية
انت مساعد ذكي مخصص للاجابة عن اسئلة المحاضرات وموجه لطلاب الهندسة المعلوماتية في جامعة حلب أجب على السؤال التالي بناءً على السياق المقدم والمحادثة السابقة. 
استخدم نفس لغة السؤال في الإجابة.
قواعد مهمة:
- إذا لم تجد المعلومات في السياق، قل: "المعلومات غير متوفرة في المستندات المرفوعة"
- استخدم نفس لغة ولهجة السؤال 
- اكتب المعادلات بشكل نصي واضح (تجنب LaTeX)
- اذا سألك عن شرح اضافي اشرح او اضف من معلوماتك ولكن لا تخرج عن سياق ومعلومات المستند
- راعِ السياق التحاوري والأسئلة السابقة
- إذا كان السؤال يشير لشيء سابق (مثل "اشرح أكثر" أو "وضّح هذا")، ارجع للمحادثة السابقة
- اذا كان السؤال عن شيئ موجود كمعلومة ولكن لم يذكر بشكل دقيق اجب من المستند ومعرفتك
- اذا اردت طرح مثال او طلب منك سؤال او مثال عن فكرة اجب بمثال او سؤال من معرفتك ولكن بالفكرة الاساسية للملف 
**السياق من المستندات:**
{context}
**المحادثة السابقة:**
{chat_history}
**السؤال الحالي:**
{question}
**الإجابة:**
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "chat_history", "question"])

retriever = None
vector_store = None
current_file_name = ""
conversation_history = [] 

def upload_file(file):
    
    global retriever, vector_store, current_file_name
    
    if file is None:
        return "❌ الرجاء اختيار ملف PDF"
    
    try:
        if not file.name.lower().endswith('.pdf'):
            return "❌ يرجى رفع ملف PDF فقط"
                
        loader = PyMuPDFLoader(file.name)
        documents = loader.load()
        
        if not documents:
            return "❌ الملف فارغ أو لا يمكن قراءته"
                
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        docs_chunks = splitter.split_documents(documents)
        
        if not docs_chunks:
            return "❌ لا يمكن تقسيم المستند"
        
        logger.info(f"تم إنشاء {len(docs_chunks)} قطعة نصية")
        
        vector_store = FAISS.from_documents(docs_chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5, 'fetch_k': 10}  
        )
        
        current_file_name = os.path.basename(file.name)
        
        return f"✅ تم تحميل الملف '{current_file_name}' بنجاح! ({len(docs_chunks)} قطعة نصية)"
        
    except Exception as e:
        return f"❌ خطأ في معالجة الملف: {str(e)}"

def format_chat_history(history):
    
    if not history:
        return "لا توجد محادثة سابقة."
    
    formatted_history = []
    
    recent_history = history[-3:] if len(history) > 3 else history
    
    for i, (q, a) in enumerate(recent_history):
        formatted_history.append(f"السؤال {i+1}: {q}")
        formatted_history.append(f"الإجابة {i+1}: {a}")
    
    return "\n".join(formatted_history)

def detect_reference_question(question, history):
    reference_keywords = [
        'اشرح أكثر', 'وضح أكثر', 'أعطني تفاصيل', 'زيد عليها', 'كمّل', 'أكمل',
        'وضح هذا', 'اشرح هذا', 'تفصيل', 'تفصيلاً', 'بالتفصيل', 'أوسع',
        'هذا الموضوع', 'هذه النقطة', 'النقطة السابقة', 'الموضوع السابق',
        'ماذا عن', 'كيف ذلك', 'لماذا', 'كيف يحدث', 'ما المقصود',
        'explain more', 'elaborate', 'tell me more', 'expand on', 'details',
        'clarify', 'what about', 'how so', 'why', 'what do you mean',
        'this topic', 'that point', 'previous', 'above', 'mentioned'
    ]
    
    question_lower = question.lower()
    
    if len(question.split()) <= 10:
        for keyword in reference_keywords:
            if keyword in question_lower:
                return True
    
    reference_pronouns = ['هذا', 'هذه', 'ذلك', 'تلك', 'this', 'that', 'it', 'they']
    for pronoun in reference_pronouns:
        if pronoun in question_lower and len(question.split()) <= 15:
            return True
    
    return False

def ask_bot(question, history):
    
    global conversation_history
    
    if not question or not question.strip():
        return history, ""
    
    if not retriever:
        error_msg = "🛑 الرجاء رفع ملف PDF أولاً قبل طرح الأسئلة"
        return history + [[question, error_msg]], ""
    
    try:        
        # تحديث تاريخ المحادثة
        conversation_history = history.copy()
        
        # تكوين تاريخ المحادثة للنموذج
        chat_history_text = format_chat_history(conversation_history)
        
        # كشف إذا كان السؤال مرجعي
        is_reference = detect_reference_question(question, conversation_history)
        
        if is_reference and conversation_history:
            logger.info("تم اكتشاف سؤال مرجعي - سيتم استخدام السياق التحاوري")
            
            enhanced_question = f"{question}\n\nملاحظة: هذا السؤال يشير للمحادثة السابقة، يرجى مراعاة السياق."
        else:
            enhanced_question = question
        
        relevant_docs = retriever.get_relevant_documents(enhanced_question)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        rag_chain = (prompt | llm | StrOutputParser())
        
        prompt_inputs = {"context": context_text,
            "chat_history": chat_history_text,
            "question": enhanced_question
        }
        
        response = rag_chain.invoke(prompt_inputs)
        
        if response:
            response = response.strip()
        else:
            response = "المعلومات غير متوفرة في المستندات المرفوعة"
        
        if is_reference:
            response = f"🔗 {response}"
        
        logger.info("تم إنتاج الإجابة بنجاح")
        new_history = history + [[question, response]]
        
        return new_history, ""
        
    except Exception as e:
        error_msg = f"❌ حدث خطأ: {str(e)}"
        return history + [[question, error_msg]], ""

def clear_chat():
    
    global conversation_history
    conversation_history = []  
    return [], ""

def get_file_info():
    
    if current_file_name and vector_store:
        return f"📄 الملف الحالي: {current_file_name}"
    return "❌ لا يوجد ملف محمل"

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AI Study Mate",
    css="""
    .gradio-container {font-family: 'Arial', sans-serif;}
    .chat-message {border-radius: 10px; padding: 10px; margin: 5px 0;}
    """
) as demo:
    
    gr.Markdown("""
    # 🎓 **Your AI Studying Mate**
    ### This AI is Developed By Osama Touma Halabi 👨‍💻 
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="💬 المحادثة",height=500,bubble_full_width=False,show_label=True)
            
            with gr.Row():
                question = gr.Textbox(placeholder="اكتب سؤالك هنا...",label="❓ سؤالك",lines=2,scale=4)
                
                send_btn = gr.Button("📤 أرسل", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ مسح المحادثة", scale=1)
                info_btn = gr.Button("ℹ️ معلومات الملف", scale=1)
        
        with gr.Column(scale=1):
            file_upload = gr.File(label="📂 ارفع ملف PDF",file_types=[".pdf"],height=200)
            
            upload_status = gr.Textbox(label="📋 حالة الرفع",interactive=False,lines=3)
            
            gr.Markdown("""
            ### 💡 نصائح:
            - ارفع ملف PDF واضح ومقروء
            - اطرح أسئلة بلغة واضحة ومفهومة
            - اطرح أسئلة محددة للحصول على إجابات دقيقة
            - يمكنك السؤال بالعربية أو الإنجليزية
            - يتذكر البوت المحادثة! يمكنك قول "اشرح أكثر" أو "وضّح هذا 
            - يحفظ آخر 3 أسئلة وإجابات للسياق
            - في حال خرج عن السياق او لم يجد معلومات كافية
            قم بسمح المحادثة وابدا من جديد
            """)
    
    send_btn.click(fn=ask_bot,inputs=[question, chatbot],outputs=[chatbot, question])
    
    question.submit(fn=ask_bot,inputs=[question, chatbot],outputs=[chatbot, question])
    
    file_upload.change(fn=upload_file,inputs=file_upload,outputs=upload_status)
    
    clear_btn.click(fn=clear_chat,outputs=[chatbot, question])
    
    info_btn.click(fn=get_file_info,outputs=upload_status)
    

if __name__ == "__main__":
    demo.launch(
        share=True,        
    )
