import os
from dotenv import load_dotenv

load_dotenv()

api_key = "oe4jAXD9zFQarh4EnX9xRNZhfPcnZZvWN-qIik0aNcCI"
ibm_cloud_url = "https://us-south.ml.cloud.ibm.com"
project_id = "1aff2ccd-0a76-4e39-90d6-b1b410716d8e"

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

db_filename = "FIA_index"
filemap_name = "filemap.pickle"
main_table = "/Users/erinhsu/Documents/GitHub/Pilot-FIA-Taiwan/tax_revenue_zh_2.csv"
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"

#for table_generator file
table_extraction_model_name = "meta-llama/llama-2-70b-chat"
tax_item_col = "稅目別"
year_col = "徵收年"
plot_name = "plot.png"

#for answer_generator file
answer_generation_model_name = "ibm-mistralai/mixtral-8x7b-instruct-v01-q"