from transformers import AutoTokenizer
import transformers
import torch, random

class FalconDistillation:
    ''' Create training samples using Falcon '''

    def __init__(self):
        self.model = "tiiuae/falcon-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.no_examples = 2
        self.prompt = open('prompt_template.txt','r').read()
        self.starters = open('prompt_conversation_starts.txt').read().split('\n')
        self.topics = open('prompt_conversation_topics.txt').read().split('\n')
    
    def generate_prompt(self):

        starter_text = ', '.join(self.starters[i] for i in random.sample(range(0, len(self.starters)), self.no_examples))
        topics_text = ', '.join(self.topics[i] for i in random.sample(range(0, len(self.topics)), self.no_examples))

        return self.prompt.format(
            conv_topics = topics_text,
            conv_starters = starter_text
        )
    
    def generate(self):

        sequences = self.pipeline(
            self.generate_prompt(),
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        print(sequences[0]['generated_text'])

dt = FalconDistillation()
dt.generate()
