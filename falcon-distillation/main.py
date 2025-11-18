from transformers import AutoTokenizer
import transformers
import torch, os, re, datetime, random

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
        self.prompt = open('prompt_template.txt','r').read()
        self.starters = open('prompt_conversation_starts.txt').read().split('\n')
        self.topics = open('prompt_conversation_topics.txt').read().split('\n')
        self.output_path = 'outputs/'
    
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
            num_return_sequences=250,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print('[+] Generated sequences, writing to file')
        fname = f'output_chat_{datetime.datetime.now().strftime("%d_%b_%Y_%H_%M")}.txt'
        with open(os.path.join(self.output_path, fname), 'w') as file:
            for seq in sequences:
                file.write('\n' + self.clean_up_text(seq['generated_text']) + '\n')

        print(f'[+] Wrote sequences to {fname}')

if __name__ == '__main__':
    dt = FalconDistillation()
    for i in range(10_000):
        dt.generate()
