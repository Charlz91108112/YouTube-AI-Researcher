import ast, random
import os, requests, json, time
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import streamlit as st

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.7)
# llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.7)

#1) Search content on Internet and get the most useful links
def search_in_google(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
    })

    headers = {
        "X-API-Key": SERPAPI_API_KEY,
        "Content-Type": "application/json",
    }

    response = rresponse = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    # with open("response.json", "w") as f:
    #     json.dump(response.json(), f)
    return response.text


# 2) Pass the JSON data to LLM and allow it to choose the best articles for it and return URLs
def find_url_for_best_artciles(response_text, query):
    print("Getting best URLs for our articles...")
    prompt = """
    You are an expert in understanding in what links are best to use for research about the following topic {query} given the data provided in the following json format:
    {response_text}. 

    After undersrtanding the 3 most relevant links to give best research output for {query} from the above json, return the list of url for all the links in the following format
    [url1, url2, url3]
    """

    prompt_template = PromptTemplate(
        input_variables=["response_text", "query"], template=prompt
    )

    article_picker_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    urls_list = article_picker_chain.predict(response_text=response_text, query=query)
    # urls_list = json.loads(urls)
    print("We got the best links:\n\n ", urls_list)

    return urls_list

# 3) Summarize the articles from the returned URLs individually and pass again to LLM to get the best output to out question
def parse_url(urls_list):
    print("Parsing the URLs for more detailed analysis...")
    urls_list = ast.literal_eval(urls_list.strip())

    loader = UnstructuredURLLoader(urls_list)
    data = loader.load()
    print("Done with the parsing...")
    return data

# 4) Summarize the content of the articles and pass it to LLM to get the best output to out question.
def summarize_articles(data, query):
    print("Summarising the articles...")
    text_splitter = CharacterTextSplitter(separator="\n", length_function=len, chunk_size=3000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)
    
    template = """
    You are a world class analyst and you know what information is important from the text, {texts}, given
    the context {query}, to create a thorough analysis report. Please follow all the below mentined rules:
    
    RULES:
    1) Make sure the content is engaging and informative.
    2) Make sure the content is very well undestood and clear.
    3) Make the sure the content includes all the imporatnt numbres and figures and not just the summary.
    4) The content is written for a professional audience.
    5) The content needs to give audience actionable advice and insights too.
    6) The fina acrticle should be in the form of a medium platform report. 
    
    SUMMARY:"""

    prompt_template = PromptTemplate(
        input_variables=["texts", "query"], template=template
    )

    summariser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    summaries = []

    for summary in enumerate(texts):
        summary = summariser_chain.predict(texts=summary, query=query)
        summaries.append(summary)

    # print("We got the best summaries:\n\n ", summaries)
    return summaries

# 4) Get the final researched data in the a structured format or generate an report for the same.
def generate_medium_report(summaries, query):
    print("Generating the final report...")
    summary_combined = "\n".join(summaries)

    template = """
    You are a world class professional and expert Information and data analyst who writes actionale reports and provide actional insights to make the technical people engaged. 
    Given the data {summary_combined} and the following content {query}, generate a well written medium article which presents thorough analysis about all the given data in a most presentable format.
    Do not add any note at the end of the article.

    ARTICLE:"""
    
    prompt_template = PromptTemplate(
        input_variables=["summary_combined", "query"], template=template
    )

    generate_article_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    article = generate_article_chain.predict(summary_combined=summary_combined, query=query)
    save_article_md(article, query)
    return article

def save_article_md(article, query):
    article_name_str = query.replace(" ", "_").replace("-", "_")
    if os.path.exists(article_name_str):
        article_name_str = article_name_str + str(random.randint(0, 100))
    with open(f"{article_name_str}.md", "w") as f:
        f.write(article)

def save_article_html(article, query):
    article_name_str = query.replace(" ", "_").replace("-", "_")
    if os.path.exists(article_name_str):
        article_name_str = article_name_str + str(random.randint(0, 100))
    with open(f"{article_name_str}.html", "w") as f:
        f.write(article)
    with open(f"{article_name_str}.txt", "w") as f:
        f.write(article)

# 5) Format the generated report and extract the required data
def format_report(article, query):
    print("Formatting the report...")

    template = """
    Given the article {article} and the context {query}, generate title of the article, 5 header sections and 5 content sections and then later populate the dictionary of the extracted data in the following format:
    1) title: Title of the article, 
    2) headers: [Header1, Header2, Header3, Header4,  Header5], 
    3) contents: [Content1, Content2, Content3, Content4, Content5]
    Make sure to use proper quotations as required to create a dictionary in python language."""

    prompt_template = PromptTemplate(
        input_variables=["article", "query"], template=template
    )

    format_report_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    formatted_report = format_report_chain.predict(article=article, query=query)
    print("Done with the formatting...")
    return ast.literal_eval(formatted_report) # returning dictionary
    

# 6) Format the generated report using HTML template
def generate_html(title, headers, contents):
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 20px;
                padding: 20px;
                background-color: #f4f4f4;
            }}

            article {{
                max-width: 800px;
                margin: 0 auto;
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}

            header h1 {{
                color: #333;
            }}

            section {{
                margin-top: 20px;
            }}

            h2 {{
                color: #007bff;
            }}

            p {{
                color: #555;
            }}
        </style>
    </head>
    <body>

        <article>
            <header>
                <h1 id="title">{title}</h1>
            </header>
    """

    for i in range(len(headers)):
        html_template += f"""
            <section>
                <h2 id="header{i + 1}">{headers[i]}</h2>
                <p id="content{i + 1}">{contents[i]}</p>
            </section>
        """

    html_template += """
        </article>

    </body>
    </html>
    """


    return html_template


st.title("Autonomous AI Researcher")
st.divider()
query = st.text_input("Enter the query")
b_container = st.container()
if query:
    with st.spinner("Generating the report..."):
        with st.spinner("Searching the Google for the best results..."):
            result = search_in_google(query)
        with st.spinner("Getting best URLs for our articles..."):
            urls_list = find_url_for_best_artciles(result, query)
            data = parse_url(urls_list)
        with st.spinner("Summarizing the articles for concise results!..."):
            summaries = summarize_articles(data, query)
        with st.spinner("Generating final report..."):
            article = generate_medium_report(summaries, query)

        st.success("Flow Completed Successfully!")

        st.divider()
        st.subheader("Final Report..")
        c = st.container(border=True)
        c.markdown(article)
        st.snow()
    # formatted_report = format_report(article, query)
    # generate_html_data = generate_html(formatted_report['title'], formatted_report['headers'], formatted_report['contents'])




# # 7) Save the generated report in HTML and text format
# save_article_html(generate_html_data, query)
# print("Article Generated Successfully!")