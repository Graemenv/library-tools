import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)
from libtools import sourceformat as sf 


#===config===
st.set_page_config(
    page_title="Coconut",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
            <style>
            #MainMenu 
            {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="collapsedControl"] {display: none}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.popover("🔗 Menu"):
    st.page_link("https://www.coconut-libtool.com/", label="Home", icon="🏠")
    st.page_link("pages/1 Scattertext.py", label="Scattertext", icon="1️⃣")
    st.page_link("pages/2 Topic Modeling.py", label="Topic Modeling", icon="2️⃣")
    st.page_link("pages/3 Bidirected Network.py", label="Bidirected Network", icon="3️⃣")
    st.page_link("pages/4 Sunburst.py", label="Sunburst", icon="4️⃣")
    st.page_link("pages/5 Burst Detection.py", label="Burst Detection", icon="5️⃣")
    st.page_link("pages/6 Keywords Stem.py", label="Keywords Stem", icon="6️⃣")
    st.page_link("pages/7 Sentiment Analysis.py", label="Sentiment Analysis", icon="7️⃣")
    st.page_link("pages/8 Shifterator.py", label="Shifterator", icon="8️⃣")
    st.page_link("pages/9 WordCloud.py", label = "WordCloud", icon = "9️⃣")
    st.page_link("pages/10 Histogram.py", label = "Histogram", icon = "🔟")


with st.expander("Before you start", expanded = True):
     
        tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download"])
        with tab1:
            st.write("The histogram will allow you to see the frequencies of specific variables in your document, such as citations, years, or specific words.")
        with tab2:
            st.text("1. Put your CSV file.")
            #more steps?

            
        with tab3:
            st.code("""
            +----------------+------------------------+--------------------------------------+
            |     Source     |       File Type        |     Column                           |
            +----------------+------------------------+--------------------------------------+
            | Scopus         | Comma-separated values | Source title,                        |
            |                | (.csv)                 | Document Type,                       |
            +----------------+------------------------| Cited by, Year                       |
            | Web of Science | Tab delimited file     |                                      |
            |                | (.txt)                 |                                      |
            +----------------+------------------------+--------------------------------------+
            | Lens.org       | Comma-separated values | Publication Year,                    |
            |                | (.csv)                 | Publication Type,                    | 
            |                |                        | Source Title,                        |
            |                |                        | Citing Works Count                   |
            +----------------+------------------------+--------------------------------------+
            | OpenAlex       | Comma-separated values | publication_year,                    |
            |                | (.csv)                 | cited_by_count,                      | 
            |                |                        | type,                                |
            |                |                        | primary_location.source.display_name |
            +----------------+------------------------+--------------------------------------+
            | Hathitrust     | .json                  | htid(Hathitrust ID)                  |
            +----------------+------------------------+--------------------------------------+
            """, language=None)          

        with tab4:  
            st.subheader(':blue[Histogram]', anchor=False)
            st.text("Click the camera icon on the top right menu (you may need to hover your cursor within the visualization)")
            st.markdown("![Downloading visualization](https://raw.githubusercontent.com/faizhalas/library-tools/main/images/download_bertopic.jpg)")
            st.subheader(":blue[Download table as CSV]", anchor=False)
            st.text("Hover cursor over table, and click download arrow")
            st.markdown("![Downloading table](https://raw.githubusercontent.com/faizhalas/library-tools/refs/heads/main/images/tablenetwork.png)")
    
st.header("Histogram Visualization", anchor=False)
st.subheader('Put your file here...', anchor=False)

#===clear cache===
def reset_all():
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def get_ext(extype):
    extype = uploaded_file.name
    return extype

@st.cache_data(ttl=3600)
def upload(extype):
    papers = pd.read_csv(uploaded_file)
    #lens.org
    if 'Publication Year' in papers.columns:
        papers.rename(columns={'Publication Year': 'Year', 'Citing Works Count': 'Cited by',
                               'Publication Type': 'Document Type', 'Source Title': 'Source title'}, inplace=True)
    elif "About the data" in papers.columns[0]:
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)
    elif "ids.openalex" in papers.columns:
        papers.rename(columns={'publication_year': 'Year', 'cited_by_count': 'Cited by',
                               'type': 'Document Type', 'primary_location.source.display_name': 'Source title'}, inplace=True)
    return papers

@st.cache_data(ttl=3600)
def conv_txt(extype):
    if("PMID" in (uploaded_file.read()).decode()): 
        uploaded_file.seek(0)
        papers = sf.medline(uploaded_file)
        print(papers)
        return papers
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DE': 'Author Keywords',
            'DT': 'Document Type',
            'AB': 'Abstract',
            'TC': 'Cited by',
            'PY': 'Year',
            'ID': 'Keywords Plus',
            'rights_date_used': 'Year'}
    uploaded_file.seek(0)
    papers = pd.read_csv(uploaded_file, sep='\t')
    if("htid" in papers.columns):
        papers = sf.htrc(papers)
    papers.rename(columns=col_dict, inplace=True)
    print(papers)
    return papers


@st.cache_data(ttl=3600)
def conv_json(extype):
    col_dict={'title': 'title',
    'rights_date_used': 'Year',
    'content_provider_code': 'Document Type',
    'Keywords':'Source title'
    }
    keywords = pd.read_json(uploaded_file)
    keywords = sf.htrc(keywords)
    keywords['Cited by'] = keywords.groupby(['Keywords'])['Keywords'].transform('size')
    keywords.rename(columns=col_dict,inplace=True)
    return keywords

def conv_pub(extype):
    if (get_ext(extype)).endswith('.tar.gz'):
        bytedata = extype.read()
        keywords = sf.readPub(bytedata)
    elif (get_ext(extype)).endswith('.xml'):
        bytedata = extype.read()
        keywords = sf.readxml(bytedata)
    keywords['Cited by'] = keywords.groupby(['Keywords'])['Keywords'].transform('size')
    st.write(keywords)
    return keywords

@st.cache_data(ttl=3600)
def readxls(file):
    papers = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
    if "About the data" in papers.columns[0]:
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)
    
    return papers

#===Read data===
uploaded_file = st.file_uploader('', type=['csv', 'txt', 'json', 'tar.gz', 'xml', 'xls', 'xlsx'], on_change=reset_all)

if uploaded_file is not None:
    try:
        extype = get_ext(uploaded_file)
        if extype.endswith('.csv'):
             papers = upload(extype) 
        elif extype.endswith('.txt'):
             papers = conv_txt(extype)
        elif extype.endswith('.json'):
            papers = conv_json(extype)
        elif extype.endswith('.tar.gz') or extype.endswith('.xml'):
            papers = conv_pub(uploaded_file)
        elif extype.endswith(('.xls', '.xlsx')):
            papers = readxls(uploaded_file)

        def get_minmax(extype):
            extype = extype
            MIN = int(papers['Year'].min())
            MAX = int(papers['Year'].max())
            MIN1 = int(papers['Cited by'].min())
            MAX1 = int(papers['Cited by'].max()) 
            unique_stitle = set()
            unique_stitle.update(papers['Source title'].dropna())
            list_stitle = sorted(list(unique_stitle))
            return papers, MIN, MAX, MIN1, MAX1, list_stitle
        
        tab1, tab2 = st.tabs(["📈 Generate visualization", "📓 Recommended Reading"])

        with tab1:
            try:
                papers, MIN, MAX, MIN1, MAX1, list_stitle = get_minmax(extype)
            except KeyError:
                st.error('Error: Please check again your columns.')
                sys.exit(1)

            stitle = st.selectbox('Focus on', (list_stitle), index=None, on_change=reset_all)
            col1, col2 = st.columns(2)
            YEAR = col1.slider('Year', min_value=MIN, max_value=MAX, value=(MIN, MAX))
            KEYLIM = col2.slider('Cited By Count',min_value = MIN1, max_value = MAX1, value = (MIN1,MAX1))
            with st.expander("Filtering settings"):
                invert_keys = st.toggle("Invert keys")
                filtered_keys = st.text_input("Filter words in source, seperate with semicolon (;)", value = "\n", on_change=None) 
                select_col = st.selectbox("Column to filter from", (list(papers)))
            keylist = filtered_keys.split(";")
            vis_choice = st.selectbox("Visualize:", ("Years", "Citation Count"))

            def listyear(extype):
                df = papers.copy()
                years = list(range(YEAR[0],YEAR[1]+1))
                cited = list(range(KEYLIM[0],KEYLIM[1]+1))
                if stitle:
                    df = df[df['Source title'].str.contains(stitle, case=False, na=False)] 
                df = df[df['Year'].isin(years)]
                df = df[df['Cited by'].isin(cited)]
                df['Cited by'] = df['Cited by'].fillna(0)
                return years, df 
            
            data = papers.copy()
            data['Cited by'] = data['Cited by'].fillna(0)

            #filtering
            if invert_keys:
                data = data[data[select_col].str.contains('|'.join(keylist), na=False)]
            else:
                data = data[~data[select_col].str.contains('|'.join(keylist), na=False)]

            
            def vis_hist(data):

                if vis_choice == "Citation Count":
                    fig = alt.Chart(pd.DataFrame(data)).mark_bar().encode(
                        x = alt.X('Cited by:Q', bin=alt.Bin(extent=[MIN1, MAX1], maxbins=20)), 
                        y = alt.Y("count()"))
                    return fig

                else:
                    fig = alt.Chart(pd.DataFrame(data)).mark_bar().encode(
                        x = alt.X('Year:Q', bin=alt.Bin(extent=[MIN, MAX], maxbins=20)),
                        y = alt.Y("count()"))
                    return fig

            years, filtered_papers = listyear(extype)
                         
            if {'Document Type','Source title','Cited by','Year'}.issubset(papers.columns):
              
                if st.button("Submit", on_click = reset_all):
                    fig = vis_hist(filtered_papers)
                    st.altair_chart(fig, use_container_width=True)


                
            else: 
                st.error('We require these columns: Document Type, Source title, Cited by, Year', icon="🚨")
        
        with tab2:
            st.markdown('**numpy.average — NumPy v1.24 Manual. (n.d.). Numpy.Average — NumPy v1.24 Manual.** https://numpy.org/doc/stable/reference/generated/numpy.average.html')
    except:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()



    

    
     
     
     
