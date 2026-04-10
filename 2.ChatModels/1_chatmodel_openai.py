from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
model = ChatOpenAI(model="gpt-5.4-nano-2026-03-17", temperature=0.3)

# Create an English to Hindi Translation Agent
def translate_english_to_hindi(text: str) -> str:
    """
    Translates English text to Hindi using OpenAI chat model.
    
    Args:
        text (str): English text to translate
        
    Returns:
        str: Translated Hindi text
    """
    messages = [
        SystemMessage(content="You are a professional English to Hindi translator. Translate the given English text to Hindi accurately while maintaining the context and meaning. Provide only the Hindi translation without any additional explanation."),
        HumanMessage(content=f"Translate the following English text to Hindi: {text}")
    ]
    
    result = model.invoke(messages)
    return result.content

# Example usage
if __name__ == "__main__":
    # Test the translation agent
    english_text = "Hello, how are you? I hope you are doing well."
    hindi_translation = translate_english_to_hindi(english_text)
    
    print("English:", english_text)
    print("Hindi:", hindi_translation)
    
    # Another example
    english_text2 = "The capital of India is New Delhi."
    hindi_translation2 = translate_english_to_hindi(english_text2)
    
    print("\nEnglish:", english_text2)
    print("Hindi:", hindi_translation2)