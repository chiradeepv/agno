"""
This example demonstrates how to use additional_input_messages with an Agent
to teach proper response patterns through few-shot learning.

Shows all supported types: str, Dict, BaseModel, Message
"""

from agno.agent import Agent
from agno.models.message import Message
from agno.models.openai.chat import OpenAIChat
from pydantic import BaseModel


class SupportResponse(BaseModel):
    """Example BaseModel for structured support responses"""
    issue_type: str
    response: str
    escalation_needed: bool = False


# Few-shot examples showing different input types
support_examples = [
    # Example 1: String input - simplest form
    "System: When a user has a password issue, always provide step-by-step instructions and be patient.",
    
    # Example 2: Dict input - like raw message format
    {
        "role": "user",
        "content": "I forgot my password and can't log in"
    },
    {
        "role": "assistant", 
        "content": """I'll help you reset your password right away.

**Steps to Reset Your Password:**
1. Go to the login page and click "Forgot Password"  
2. Enter your email address
3. Check your email for the reset link
4. Follow the link to create a new password
5. Use a strong password with uppercase, lowercase, numbers, and symbols

If you don't receive the email within 5 minutes, please check your spam folder. Let me know if you need any assistance with these steps!"""
    },
    
    # Example 3: BaseModel input - structured data
    SupportResponse(
        issue_type="billing_error",
        response="""I sincerely apologize for the billing error and the frustration this has caused you.

**Immediate Action Plan:**
1. I'll investigate your account to confirm the duplicate charge
2. Process a full refund for the duplicate transaction  
3. Ensure this doesn't happen again by checking your payment method settings
4. Provide you with a confirmation number once the refund is processed

The refund typically takes 3-5 business days to appear on your statement. As an apology for this inconvenience, I'd also like to offer you a 10% discount on your next order.""",
        escalation_needed=False
    ),
    
    # Example 4: Message input - full Message object  
    Message(role="user", content="Your app keeps crashing when I try to upload photos"),
    Message(
        role="assistant",
        content="""I'm sorry you're experiencing crashes with photo uploads. Let me help you troubleshoot this issue.

**Troubleshooting Steps:**
1. **Check App Version:** Ensure you have the latest version from the app store
2. **Restart the App:** Close completely and reopen the application  
3. **Check Storage:** Make sure you have sufficient device storage (at least 1GB free)
4. **Photo Size:** Try uploading smaller photos (under 10MB each)
5. **Network Connection:** Ensure you have a stable internet connection

**If the issue persists:**
- Try uploading one photo at a time instead of multiple
- Clear the app cache in your device settings
- Restart your device

If none of these steps resolve the issue, please let me know your device type and OS version, and I'll escalate this to our technical team for further investigation."""
    ),
]

if __name__ == "__main__":
    # Create agent with few-shot learning using different input types
    agent = Agent(
        name="Customer Support Specialist",
        model=OpenAIChat(id="gpt-4o-mini"),
        add_name_to_context=True,
        additional_input_messages=support_examples,  # Demonstrates all input types
        instructions=[
            "You are an expert customer support specialist.",
            "Always be empathetic, professional, and solution-oriented.",
            "Provide clear, actionable steps to resolve customer issues.",
            "Follow the established patterns for consistent, high-quality support.",
        ],
        debug_mode=True,
        markdown=True,
    )

    print("🔍 Demonstrating different additional_input_messages types:")
    print("=" * 60)
    
    for i, example in enumerate(support_examples, 1):
        example_type = type(example).__name__
        if isinstance(example, dict):
            example_type = "Dict"
        elif isinstance(example, str):
            example_type = "String"
        elif isinstance(example, SupportResponse):
            example_type = "BaseModel (SupportResponse)"
        elif isinstance(example, Message):
            example_type = "Message"
            
        print(f"📝 Example {i} ({example_type}): {example}")
        print("-" * 50)
    
    print("\n🤖 Now testing the agent with few-shot learning:")
    print("=" * 60)
    
    # Test the agent with various support scenarios
    test_scenarios = [
        "I can't remember my account email address",
        "The website charged my card but I never received my order", 
        "My mobile app won't sync with the web version"
    ]
    
    for scenario in test_scenarios:
        print(f"\n💬 User Query: {scenario}")
        print("-" * 50)
        agent.print_response(scenario)
