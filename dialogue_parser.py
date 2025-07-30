import re

class DialogueParser:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding
        
    def extract_rounds(self, content):
        """Extract rounds from the dialogue content using Round markers."""
        round_pattern = r'Round(\d+)'
        round_splits = re.split(round_pattern, content)
        
        rounds_dict = {}
        for i in range(1, len(round_splits), 2):
            if i+1 < len(round_splits):
                round_number = round_splits[i]
                round_content = round_splits[i+1].strip()
                rounds_dict[f"Round{round_number}"] = round_content
        
        return rounds_dict
    
    def parse_turns(self, round_content):
        """Parse the turns (User/Agent exchanges) in a round."""
        turns = []
        lines = round_content.split('\n')
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("User: "):
                # Save previous turn if exists
                if current_speaker and current_text:
                    turns.append({
                        "speaker": current_speaker,
                        "text": '\n'.join(current_text).strip()
                    })
                
                # Start new User turn
                current_speaker = "User"
                current_text = [line[6:]]  # Remove "User: " prefix
            elif line.startswith("Agent: "):
                # Save previous turn if exists
                if current_speaker and current_text:
                    turns.append({
                        "speaker": current_speaker,
                        "text": '\n'.join(current_text).strip()
                    })
                
                # Start new Agent turn
                current_speaker = "Agent"
                current_text = [line[7:]]  # Remove "Agent: " prefix
            elif line:
                # Continue with current turn
                current_text.append(line)
        
        # Add the last turn
        if current_speaker and current_text:
            turns.append({
                "speaker": current_speaker,
                "text": '\n'.join(current_text).strip()
            })
        
        return turns
    
    def parse_dialogue(self, content):
        """Parse the complete dialogue content into structured data."""
        rounds_dict = self.extract_rounds(content)
        
        structured_dialogue = {}
        for round_name, round_content in rounds_dict.items():
            structured_dialogue[round_name] = self.parse_turns(round_content)
        
        return structured_dialogue
    
    def extract_rounds_subset(self, structured_dialogue, max_rounds):
        """Extract only a subset of rounds (1 to max_rounds)."""
        subset = {}
        for i in range(1, max_rounds + 1):
            round_key = f"Round{i}"
            if round_key in structured_dialogue:
                subset[round_key] = structured_dialogue[round_key]
        return subset
    
    def format_dialogue_for_llm(self, rounds_data):
        """Format dialogue rounds for LLM input."""
        formatted_text = ""
        for round_name, turns in sorted(rounds_data.items(), key=lambda x: int(x[0][5:])):
            formatted_text += f"\n{round_name}\n"
            for turn in turns:
                formatted_text += f"{turn['speaker']}: {turn['text']}\n"
        return formatted_text.strip()


if __name__ == "__main__":
    # Example usage
    parser = DialogueParser()
    example_content = """
    Round1
    User: Hello, how are you?
    Agent: I'm fine, thank you! How can I assist you today?
    Round2
    User: What is the weather like?
    Agent: It's sunny and warm today.
    """
    structured_dialogue = parser.parse_dialogue(example_content)
    print("Structured Dialogue:\n", structured_dialogue)
    subset_dialogue = parser.extract_rounds_subset(structured_dialogue, 6)  # Fixed to 6 rounds
    print("Subset Dialogue:\n", subset_dialogue)
    formatted_text = parser.format_dialogue_for_llm(subset_dialogue)
    print("Formatted Text for LLM:\n", formatted_text)
