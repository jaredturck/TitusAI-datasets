from transformers import AutoTokenizer
import transformers
import torch, random

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
    
    def generate_prompt(self):

        starter_text = random.choice(self.starters)
        topics_text = random.choice(self.topics)

        return f'{self.prompt} for example {starter_text} {topics_text}'
    
    def generate(self):

        sequences = self.pipeline(
            self.generate_prompt(),
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=1.2,
            num_return_sequences=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            max_length=2048,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with open('output_chat.txt', 'w') as file:
            for seq in sequences:
                file.write('\n[BOS]\n' + seq['generated_text'] + '\n[EOS]\n')

        print(sequences[0]['generated_text'])

dt = FalconDistillation()
dt.generate()
