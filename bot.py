from typing import Final
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler,Application, MessageHandler,filters

import tensorflow as tf
from keras.models import load_model
from keras.layers import TextVectorization
import numpy as np
import pandas as pd
import creds
import re

#token and username
TOKEN: Final = creds.TOKEN
BOT_USERNAME: Final = '@ToxiccommentBot'

#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Hello! {update.message.from_user.first_name}. Please add me to your group😊')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Type a comment and I will tell you if it is toxic or not')

#load model
model=load_model('toxic_text_detector50.h5')

#load tokenizer
def ret_x():
    tc=pd.read_csv('train.csv')
    X = tc['comment_text'].values
    return X

Y=['toxic','severe_toxic','obscene','threatning','insulting','identity_hate']
tokenizer = TextVectorization(output_sequence_length=3000)
tokenizer.adapt(ret_x())

#preict toxicity
def predict_toxicity(text: str) -> list:

    predict_comment=tokenizer(text)
    prediction = model.predict(np.array([predict_comment]))

    return ((prediction> 0.5).astype(bool)).tolist()[0]

# Clean Comments Function that gets text
def clean_comments(text):
    
    # Lowercase comment
    text = text.lower()
    
    # Replace words to clean
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    
    # Erase irrelevant characters
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('\\"', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    
    # Strip the sentence (remove first and last space character)
    text = text.strip(' ')
    
    # Return the cleaned text
    return text

#Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    text = str(update.message.text).lower()

    res = predict_toxicity(clean_comments(text))
    
    score=sum(res)
    #check if toxic
    
    if score > 1:
        response=f"""{update.message.from_user.first_name} Your message has been flagged by our system for containing toxic language🔴"""

        await update.message.reply_text(response)
    


#main
if __name__ == '__main__':

    print("started")
    app = Application.builder().token(TOKEN).build()

    #Commands_Handler
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    #Message_Handler
    app.add_handler(MessageHandler(filters=filters.TEXT, callback=handle_message))

    app.run_polling(poll_interval=0.5)
    


