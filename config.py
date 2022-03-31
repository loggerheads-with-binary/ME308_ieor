import os 
import gspread
from oauth2client.service_account import ServiceAccountCredentials

PROG_PATH = os.path.dirname(os.path.abspath(__file__))

CLIENT_SECRETS = os.path.join(PROG_PATH , 'credentials' , 'client_secrets.json')
STORAGE_FILE = os.path.join(PROG_PATH , 'credentials' , 'storage.json')

SCOPE = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
 

def sheets_init():

    global SCOPE , CLIENT_SECRETS, STORAGE_FILE 
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(CLIENT_SECRETS , SCOPE)
    client = gspread.authorize(creds)
    return client 
   
