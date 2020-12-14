from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import pipeline


import pandas as pd
data = pd.read_csv('goroscope.csv')

nl = []
for i in data['znak']:
  nl.append(i)
  
  
def build_text_files(txts, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for txt in txts:
        summary = txt.strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)
    
train, test = train_test_split(nl[:50], test_size=0.15)

build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3medium_based_on_gpt2')

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

def load_dataset(train_path,test_path,tokenizer):

    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=16)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=16)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,

    )
    return train_dataset,test_dataset,data_collator

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

training_args = TrainingArguments(
    output_dir="./rugpt3", 
    overwrite_output_dir=True, 
    num_train_epochs=3, 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,  
    eval_steps = 400, 
    save_steps=800, 
    warmup_steps=500
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model()

text_generation = pipeline('text-generation', model='./rugpt3', tokenizer='sberbank-ai/rugpt3medium_based_on_gpt2', config={'max_length':1000, 'temperature': .5})
text = "Скоро вы узнаете, что"

result = text_generation(text, max_length=150, min_length=50, do_sample=True, temperature=1.2)[0]['generated_text']
print(result)
speech
