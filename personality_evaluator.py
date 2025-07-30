import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import concurrent.futures



class PersonalityEvaluator:
    def __init__(self, api_key=None, api_base=None, model_name="gpt-3.5-turbo", temperature=0.0):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LangChain components if API key is provided
        if api_key:
            os.environ["OPENAI_API_BASE"] = api_base
            os.environ["OPENAI_API_KEY"] = api_key
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=temperature
            )
        else:
            self.chat_model = None
        
        # 移除profile相关内容
        self.prompt_template = PromptTemplate(
            input_variables=["dialogue"],
            template="""
### Background:  
You are a professional personality psychologist specializing in the Big Five personality traits model. You've been invited to analyze the personality traits of a human player in a "Prisoner's Dilemma" game. In this game, the human player competes against an AI agent, with each round consisting of two phases: dialogue and decision-making, where players can choose to "cooperate" or "betray."

### Task:
1. You are to analyze the human player's personality traits based on Game Dialogue Record. You will provide a detailed analysis of each of the Big Five personality traits, including specific examples from the dialogue to support your ratings.
2. Your response should strictly follow the Response Template.

### Big Five Personality Traits Reference Standards: 
#### Openness: 
- High Scores (3.1-5.0): Curious, imaginative, creative, open to trying new things, unconventional thinking
- Medium Scores (3.0): Maintains balance between tradition and innovation, shows some curiosity while also valuing stability
- Low Scores (1.0-2.9): Predictable, not very imaginative, resistant to change, prefers routine, traditional thinking

#### Conscientiousness: 
- High Scores (3.1-5.0): Competent, organized, dutiful, achievement-striving, self-disciplined, deliberate
- Medium Scores (3.0): Shows some planning and responsibility while maintaining some flexibility
- Low Scores (1.0-2.9): Incomplete, disorganized, careless, procrastinates, lacks self-discipline, impulsive

#### Extraversion: 
- High Scores (3.1-5.0): Sociable, energized by social interaction, excitement-seeking, enjoys being the center of attention, outgoing
- Medium Scores (3.0): Balances social interaction and solitude, situational social behavior
- Low Scores (1.0-2.9): Prefers solitude, fatigued by excessive social interaction, reflective, dislikes being the center of attention, reserved

#### Agreeableness: 
- High Scores (3.1-5.0): Trusting (forgiving), straightforward, altruistic (enjoys helping), compliant, modest, sympathetic, empathetic
- Medium Scores (3.0): Selectively shows friendliness based on situations, balances cooperation and self-interest
- Low Scores (1.0-2.9): Skeptical, demanding, insults and belittles others, stubborn, show-off, unsympathetic, doesn't care about others' feelings

#### Neuroticism: 
- High Scores (3.1-5.0): Anxious, hostile anger (irritable), frequently stressed, self-conscious (shy), vulnerable, experiences dramatic mood shifts
- Medium Scores (3.0): Moderate emotional fluctuations, relatively stable under pressure
- Low Scores (1.0-2.9): Doesn't worry much, calm, emotionally stable, confident, resilient, rarely feels sad or depressed

### Rating Criteria:
1.0-1.9: Very low - Rarely if ever displays characteristics associated with this trait
2.0-2.7: Low - Occasionally displays characteristics associated with this trait
2.8-3.2: Average - Shows balanced or moderate expression of this trait
3.3-4.0: High - Frequently displays characteristics associated with this trait
4.1-5.0: Very high - Strongly and consistently displays characteristics associated with this trait

### Boundary Value Handling:
- All intervals are closed intervals, meaning they include the endpoint values
- The handling of boundary values 1.0, 1.9, 2.0, 2.7, 2.8, 3.2, 3.3, 4.0, 4.1, and 5.0 is as follows:
  - 1.0 ≤ score ≤ 1.9: Classified as "Very low"
  - 2.0 ≤ score ≤ 2.7: Classified as "Low"
  - 2.8 ≤ score ≤ 3.2: Classified as "Average"
  - 3.3 ≤ score ≤ 4.0: Classified as "High"
  - 4.1 ≤ score ≤ 5.0: Classified as "Very high"

- Decimal precision explanation (e.g., 2.3, 3.7, 4.5):
  - Lower decimals within each range (e.g., 3.3-3.5) indicate emerging or inconsistent expression
  - Middle decimals (e.g., 3.6-3.7) indicate moderate expression within that range
  - Higher decimals (e.g., 3.8-4.0) indicate strong expression approaching the next level


### Analysis Requirements: 
1. Carefully read the entire dialogue record, paying special attention to the human player's decision patterns, communication style, and emotional expression.
2. Rate the human player on each dimension of the Big Five personality traits on a scale of 1-5.
3. Base your ratings on specific evidence from the dialogue, avoiding subjective assumptions.
4. Quote original text from the dialogue as supporting evidence in your analysis.
5. Provide at least 2-3 specific examples as the basis for each dimension's rating.
6. Think step by step, finding evidence before drawing conclusions.
7. Ensure balanced analysis by considering both positive and negative expressions of the same trait.

### Important Format Instructions
1) For each trait, you must start a new line in the format:
- Openness: X, reason: ...
- Conscientiousness: X, reason: ...
- Extraversion: X, reason: ...
- Agreeableness: X, reason: ...
- Neuroticism: X, reason: ...
Where `X` is a single integer or a float from 1-5 (e.g. 4.0, 3.7, 2.3), and then a comma, then ` reason:`.  

### Response Template: 
### My step by step thought process: 
{{Detailed explanation of how you analyzed each dimension, including key behaviors and dialogue you noticed}} 

### Player's Personality Traits Rating: 
- Openness: {{Rating}}, reason: {{Detailed analysis based on specific dialogue content, at least 2-3 examples}} 
- Conscientiousness: {{Rating}}, reason: {{Detailed analysis based on specific dialogue content, at least 2-3 examples}} 
- Extraversion: {{Rating}}, reason: {{Detailed analysis based on specific dialogue content, at least 2-3 examples}} 
- Agreeableness: {{Rating}}, reason: {{Detailed analysis based on specific dialogue content, at least 2-3 examples}} 
- Neuroticism: {{Rating}}, reason: {{Detailed analysis based on specific dialogue content, at least 2-3 examples}}

### Game Dialogue Record: 
{dialogue}
"""
        )
        
    def batch_evaluate_personality(self, tasks, ablation_choice='dialogue'):
        """并发评估多个用户人格任务"""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for task in tasks:
                dialogue_text = task['dialogue']
                future = executor.submit(self.evaluate_personality, dialogue_text, ablation_choice)
                futures.append((future, task))
            
            for future, task in futures:
                try:
                    result = future.result()
                    results.append((task, result))
                except Exception as e:
                    print(f"Error in batch evaluation: {e}")
                    results.append((task, self._get_mock_evaluation()))
        return results
    
    def evaluate_personality(self, dialogue_text, ablation_choice='dialogue'):
        """对对话进行评估，若LLM未配置，则返回Mock。
        
        ablation_choice参数仅保留'dialogue'参数以兼容旧接口
        """
        if not self.chat_model:
            print("No valid LLM configured, returning mock result.")
            return self._get_mock_evaluation()
            
        try:
            chain = LLMChain(prompt=self.prompt_template, llm=self.chat_model)
            response = chain.invoke({"dialogue": dialogue_text})
            
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response)
            
            parsed = self._parse_llm_response(response_text)
            return parsed
        except Exception as e:
            print(f"Exception during LLM evaluation: {e}")
            return self._get_mock_evaluation()
        
    def _parse_llm_response(self, response):
        """
        强化版正则解析：
        - 分别匹配5个Big Five维度
        - 若全部维度都为None，则返回Mock
        - 部分维度成功则保留其值，未匹配到的为None
        """
        # 为每个维度编写多行匹配的正则
        # 例如: "- Openness: 4, reason: 这里是一大段换行描述..."
        trait_patterns = {
            "O": re.compile(
                r'-\s*Openness\s*:\s*([\d\.]+)\s*,\s*reason\s*:\s*(.*?)(?=\n-\s*Conscientiousness|###|$)',
                re.DOTALL
            ),
            "C": re.compile(
                r'-\s*Conscientiousness\s*:\s*([\d\.]+)\s*,\s*reason\s*:\s*(.*?)(?=\n-\s*Extraversion|###|$)',
                re.DOTALL
            ),
            "E": re.compile(
                r'-\s*Extraversion\s*:\s*([\d\.]+)\s*,\s*reason\s*:\s*(.*?)(?=\n-\s*Agreeableness|###|$)',
                re.DOTALL
            ),
            "A": re.compile(
                r'-\s*Agreeableness\s*:\s*([\d\.]+)\s*,\s*reason\s*:\s*(.*?)(?=\n-\s*Neuroticism|###|$)',
                re.DOTALL
            ),
            "N": re.compile(
                r'-\s*Neuroticism\s*:\s*([\d\.]+)\s*,\s*reason\s*:\s*(.*?)(?=###|$)',
                re.DOTALL
            )
        }
        
        trait_ratings = {}
        trait_reasons = {}
        
        for trait_key, pattern in trait_patterns.items():
            match = pattern.search(response)
            if match:
                rating_str = match.group(1).strip()
                reason_str = match.group(2).strip()
                try:
                    rating_val = float(rating_str)
                    trait_ratings[trait_key] = rating_val
                    trait_reasons[trait_key] = reason_str
                except ValueError:
                    trait_ratings[trait_key] = None
                    trait_reasons[trait_key] = f"Parse rating failed: {rating_str}"
            else:
                trait_ratings[trait_key] = None
                trait_reasons[trait_key] = "Not found"
        
        # 抽取思考过程
        thought_match = re.search(r'###\s*My step by step thought process\s*:\s*(.*)', response, re.DOTALL)
        thought_process = thought_match.group(1).strip() if thought_match else ""
        
        # 如果5个维度全为None，返回Mock
        if all(v is None for v in trait_ratings.values()):
            return self._get_mock_evaluation()
        
        return {
            "ratings": trait_ratings,
            "reasons": trait_reasons,
            "thought_process": thought_process,
            "raw_response": response
        }
    
    def _get_mock_evaluation(self):
        """完全无法解析时返回的默认值"""
        return {
            "ratings": {
                "O": 3,
                "C": 4,
                "E": 3,
                "A": 2,
                "N": 3
            },
            "reasons": {
                "O": "Mock reason (Openness)",
                "C": "Mock reason (Conscientiousness)",
                "E": "Mock reason (Extraversion)",
                "A": "Mock reason (Agreeableness)",
                "N": "Mock reason (Neuroticism)"
            },
            "thought_process": "Mock step by step process.",
            "raw_response": "Mock response"
        }

    

if __name__ == "__main__":
    # Example usage
    evaluator = PersonalityEvaluator(api_key="your-api-key", api_base="your-api-base")
    example_content = """
    Round1
    User: Hello, how are you?
    Agent: I'm fine, thank you! How can I assist you today?
    User: What is the weather like?
    Agent: It's sunny and warm today.
    """
    result = evaluator.evaluate_personality(example_content)
    print(result)