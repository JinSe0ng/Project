import openai
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
import pymysql
import sys
import locale

# .env 파일에서 환경 변수 로드
load_dotenv(r'C:\sesac_lagchain\.env')  # 경로가 올바른지 확인

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

####################################
####### sql query 생성 함수 #######
####################################

db_uri = 'mysql+pymysql://admin:sesaclangchain@database-1.cxcqeqcc6xxo.ap-northeast-2.rds.amazonaws.com:3306/sesaclangchain'
create_sql_query_db = SQLDatabase.from_uri(db_uri)  # ✅ 모든 테이블 포함

# ✅ 모든 테이블 목록 가져오기
def get_all_tables():
    return create_sql_query_db.get_usable_table_names()  # ✅ 사용 가능한 테이블 목록 반환

# ✅ 질문과 관련된 테이블만 선택 (강의 테이블과 리뷰 테이블 필수 포함)
def get_relevant_tables(question):
    table_names = get_all_tables()  # ✅ 모든 테이블 가져오기
    essential_tables = {"offline_course_tbl", "review"}  # ✅ 필수적으로 포함해야 하는 테이블

    prompt = ChatPromptTemplate.from_template("""
    Given the database tables: {tables}, which tables are most relevant to answer the question: "{question}"?
    Respond with a comma-separated list of relevant table names.
    """)
    llm = ChatOpenAI(model="gpt-4", max_tokens=50)
    
    relevant_tables = llm.invoke(prompt.format(tables=", ".join(table_names), question=question))
    
    # ✅ 'AIMessage' 객체에서 .content 속성을 가져오기
    if hasattr(relevant_tables, "content"):
        relevant_tables = relevant_tables.content.strip()
    
    # ✅ 필수 테이블 추가 후 중복 제거
    relevant_tables = set(relevant_tables.split(", ")) | essential_tables
    return list(relevant_tables)  # ✅ LLM이 선택한 테이블 목록 반환


# ✅ 선택된 테이블의 스키마만 가져오기
def get_schema(question):
    relevant_tables = get_relevant_tables(question)  # ✅ 관련 테이블 찾기
    return create_sql_query_db.get_table_info(relevant_tables)  # ✅ 해당 테이블들의 스키마 반환

def create_sql(question):
    create_sql_query_template = """
    Based on the table schema below, write only the SQL query that would answer the user's question.
    Do not include any explanation, comments, or additional text. Return only the SQL query.

    Make sure to join tables properly using course_id instead of course_title if available.

    {schema}

    Question: {question}

    SQL Query:
    """

    create_sql_query_prompt = ChatPromptTemplate.from_template(template=create_sql_query_template)

    # ✅ OpenAI 모델 사용
    llm = ChatOpenAI(model="gpt-4", max_tokens=256)

    # ✅ SQL 체인 실행
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)  # ✅ 질문 기반으로 필요한 테이블 선택
        | create_sql_query_prompt  # ✅ 프롬프트 생성
        | llm  # ✅ SQL 쿼리 생성
        | StrOutputParser()  # ✅ 결과 파싱
    )

    # ✅ 실행

    result = sql_chain.invoke({"question": question})
    sql_query = result.strip("```sql\n").strip("```")
    print(sql_query)
    sys.stdout.flush()
    return sql_query


####################################
#### sql 토대로 db 갖고오는 함수 ####
####################################
def connect_to_db():
    return pymysql.connect(
        host=os.getenv("HOST_NAME"),
        port=3306,
        user='admin',
        passwd=os.getenv("PASSWORD"),
        db='sesaclangchain',
        charset="utf8mb4"
    )

def fetch(sql_query):
    connection = connect_to_db()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    cursor.execute(sql_query)
    fetch_result = cursor.fetchall()

    if not fetch_result:
        print("No results found for the query.")

    connection.close()

    return fetch_result


def convert_to_documents(courses):
    documents = []

    # 모든 강의를 문서로 변환
    for course in courses:
        doc_text = ""
        for key, value in course.items():
            # 각 필드를 '필드명: 값' 형태로 문서화
            doc_text += f"{key}: {value}\n"
        documents.append(Document(page_content=doc_text))

    return documents

# 벡터 스토어 생성 (FAISS 사용)
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# 질문과 관련된 강의를 검색하고 답변을 생성하는 챗봇
def create_chatbot(vector_store, user_prompt, system_prompt):
    # OpenAI LLM 설정
    llm = OpenAI(temperature=0)
    
    # 프롬프트 템플릿 생성
    prompt_template = """
    시스템 프롬프트: {system_message}
    사용자 프롬프트: {user_message}
    사용자의 질문: {question}
    """
    
    prompt = PromptTemplate(
        input_variables=["system_message", "user_message", "question"],
        template=prompt_template
    )

    # PromptTemplate을 사용하여 LLM을 설정
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # ConversationalRetrievalChain 생성
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model, 
        retriever=vector_store.as_retriever(), 
        return_source_documents=True
    )

    return retrieval_chain


# 사용자 질문에 대한 답변을 생성하는 함수
def get_answer(question, retrieval_chain, chat_history):
    # 질문을 `question` 키와 `chat_history` 키를 포함한 딕셔너리 형태로 전달
    result = retrieval_chain({"question": question, "chat_history": chat_history})

    # 'answer'와 'source_documents' 두 값을 받으므로, 'answer'만 반환하도록 처리
    answer = result.get('answer', '')
    source_documents = result.get('source_documents', None)  # source_documents가 없을 수 있으므로 처리

    return answer, source_documents

def main():
    # 시스템 로케일을 'ko_KR.UTF-8'로 설정하여 한글 깨짐을 방지
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

    # 사용자로부터 질문 입력 받기
    question = input("질문을 입력하세요: ")

    # SQL 쿼리 생성
    sql_query = create_sql(question)

    # 강의 데이터 가져오기
    courses = fetch(sql_query)

    # 강의 정보를 문서로 변환
    documents = convert_to_documents(courses)

    # 벡터 스토어 생성
    vector_store = create_vector_store(documents)

    # 사용자 프롬프트와 시스템 프롬프트 정의
    system_prompt = "당신은 싹가능이 만든 한국의 도시 서울에서 제공하는 청년취업사관학교 챗봇 싹톡입니다. 당신의 임무는 주어진 데이터를 토대로 정보를 알려주세요."
    user_prompt = "당신에게 주어진 정보는 교육프로그램과 관련된 데이터, 공지사항에 관한 정보를 가지고 있으며, 그 외 질문에는 답변이 어렵다고 말하세요."

    # 챗봇 생성
    retrieval_chain = create_chatbot(vector_store, user_prompt, system_prompt)

    # 대화 기록 초기화
    chat_history = []

    # 사용자 질문 받기
    while True:
        question = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")

        if question.lower() == 'exit':
            break

        # 답변 생성
        answer, source_documents = get_answer(question, retrieval_chain, chat_history)

        # 답변 출력 (한글이 깨지지 않도록)
        print(f"질문: {question}")
        sys.stdout.flush() # 즉시 출력하게 하는함수
        print(f"답변: {answer}")
        sys.stdout.flush()

        # chat_history에 현재 질문과 답변을 추가
        chat_history.append((question, answer))


if __name__ == "__main__":
    main()
