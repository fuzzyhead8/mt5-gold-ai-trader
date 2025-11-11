"""
Test script to check if OpenAI and Anthropic API keys are working
"""

import os
import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_api():
    """Test OpenAI API connection"""
    print("🔄 Testing OpenAI API...")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    try:
        client = openai.OpenAI(api_key=api_key)

        # Simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, test message"}],
            max_tokens=10
        )

        if response.choices[0].message.content:
            print("✅ OpenAI API working")
            return True
        else:
            print("❌ OpenAI API returned empty response")
            return False

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_anthropic_api():
    """Test Anthropic API connection"""
    print("🔄 Testing Anthropic API...")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Simple test call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello, test message"}]
        )

        if response.content[0].text:
            print("✅ Anthropic API working")
            return True
        else:
            print("❌ Anthropic API returned empty response")
            return False

    except Exception as e:
        print(f"❌ Anthropic API error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing AI API Connections")
    print("=" * 40)

    openai_ok = test_openai_api()
    anthropic_ok = test_anthropic_api()

    print("\n" + "=" * 40)
    if openai_ok and anthropic_ok:
        print("🎉 All AI APIs are working!")
    elif openai_ok:
        print("⚠️ OpenAI API working, Anthropic API failed")
    elif anthropic_ok:
        print("⚠️ Anthropic API working, OpenAI API failed")
    else:
        print("❌ Both AI APIs failed")
