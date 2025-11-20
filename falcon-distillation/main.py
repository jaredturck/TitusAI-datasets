from transformers import AutoTokenizer
import transformers
import torch, os, re, datetime, random, requests

STATUS_WEBHOOK = 'https://discord.com/api/webhooks/1431466888956870677/bg5j5IZiG95bqsgQngre_JZm74MtXtgNCcrA_Q7Xe2mTuJ7lxTHe65jYMyJKPvw_Jq2H'

def send_status(message):
    ''' Sends a status update to the Discord webhook '''
    try:
        requests.post(STATUS_WEBHOOK, json={'content': message})
        print(message)
    except Exception as e:
        print(f'[error] Failed to send status update: {e}')

class FalconDistillation:
    ''' Create training samples using Falcon '''

    def __init__(self):
        self.model = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.prompt = open('prompt_template_flirt.txt','r').read()
        self.starters = open('prompt_conversation_starts.txt').read().split('\n')
        self.topics = open('prompt_conversation_topics.txt').read().split('\n')
        self.output_path = 'outputs_dataset_3/'

    def generate_prompt(self):

        starter_text = random.choice(self.starters)
        topics_text = random.choice(self.topics)

        return f'{self.prompt} for example {starter_text} {topics_text}'
    
    def clean_up_text(self, text):
        txt = '\n'.join(filter(lambda x : len(x) >= 3, text.split('\n')))
        txt = re.sub(r'^(person|person a|person b|person one|person two|friend|user|you|bot)[\s]*[\d+:\s]*', '', txt, flags=re.IGNORECASE|re.MULTILINE)
        txt = re.sub(r'^\d+[\.\):]*', '', txt, flags=re.IGNORECASE|re.MULTILINE)
        return txt

    def generate(self):
        ''' Dataset 1 and 2 '''
        seed = random.randrange(0, 2**32-1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(f'[+] Generating sequences with seed {seed}')
        sequences = self.pipeline(
            self.generate_prompt(),
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=150,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print('[+] Generated sequences, writing to file')
        fname = f'output_chat_{datetime.datetime.now().strftime("%d_%b_%Y_%H_%M")}_{seed}.txt'
        with open(os.path.join(self.output_path, fname), 'w') as file:
            for seq in sequences:
                file.write('\n' + self.clean_up_text(seq['generated_text']) + '\n')

        send_status(f'[+] Wrote sequences to {fname}')
    
    def generate_2(self):
        ''' Dataset 3 '''
        self.output_path = 'outputs_dataset_3/'
        try:
            for n,prompt in enumerate(self.topics + self.starters):
                if n < 580:
                    print(f'[+] Skipped prompt {prompt}')
                    continue

                seed = random.randrange(0, 2**32-1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                print(f'[+] prompt: {prompt}')
                sequences = self.pipeline(
                    prompt,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8,
                    num_return_sequences=200,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                    return_full_text=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                fname = f'output_chat_{datetime.datetime.now().strftime("%d_%b_%Y_%H_%M")}_{seed}.txt'
                with open(os.path.join(self.output_path, fname), 'a', encoding='utf-8') as file:
                    for seq in sequences:
                        file.write('\n' + self.clean_up_text(seq['generated_text']) + '\n')

                send_status(f'[+] Wrote sequences to {fname}')
        except Exception as e:
            send_status(f'[error] Exception during generation: {e}')
    
    def process_pipeline(self, prompt, num_return_sequences):
        seed = random.randrange(0, 2**32-1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f'[+] prompt: {prompt}')
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=num_return_sequences,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            return_full_text=True,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        fname = f'output_chat_{datetime.datetime.now().strftime("%d_%b_%Y_%H_%M")}_{seed}.txt'
        with open(os.path.join(self.output_path, fname), 'a', encoding='utf-8') as file:
            for seq in sequences:
                file.write('\n' + self.clean_up_text(seq['generated_text']) + '\n')

        send_status(f'[+] Wrote sequences to {fname}')
    
    def generate_3(self):
        ''' Dataset 4 '''
        self.output_path = 'outputs_dataset_4/'
        questions = open('prompt_questions.txt').read().split('\n')
        open_ended_questions = open('prompt_open_ended_questions.txt').read().split('\n')
        try:
            for prompt in questions:
                self.process_pipeline(prompt, num_return_sequences=50) # 50 samples lower quality questions
            
            for prompt in open_ended_questions:
                self.process_pipeline(prompt, num_return_sequences=200) # 200 samples higher quality open ended questions
                
        except Exception as e:
            send_status(f'[error] Exception during generation: {e}')

if __name__ == '__main__':
    dt = FalconDistillation()
    # dt.generate() - dataset 1 and 2
    dt.generate_2() # dataset 3
    dt.generate_3() # dataset 4
