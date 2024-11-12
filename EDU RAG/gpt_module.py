import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict
import re

load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

def generate_prompt(content, dok_level):
    """
    Generates a detailed prompt for the GPT model based on the educational content and DOK level.

    Args:
    content (str): Educational content or topic from which to generate a math problem.
    dok_level (int): Desired Depth of Knowledge level (1-4).

    Returns:
    str: A fully formed prompt including detailed instructions and examples.
    """
    
    # Define DOK level details
    dok_details = {
        1: {
            'description': 'Mathematical Recall and Reproduction: Tasks at this level require students to recall facts, definitions, or procedures.',
            'examples': 'Apply a well-known algorithm, Identify a plane or three-dimensional figure, Perform a specified or routine procedure.',
            'math_example': 'The price of gasoline was $2.159 per gallon last week. This week the new price is $2.319 per gallon. Determine the percent of increase.',
            'reasoning': 'Students will identify a transformation within a plane.'
        },
        2: {
            'description': 'Mathematical Skills and Concepts: Tasks at this level involve some mental processing beyond recalling or reproducing a response.',
            'examples': 'Solve a routine problem requiring multiple steps, or the application of multiple concepts, interpreting data, explaining relationships between concepts.',
            'math_example': 'On a trip across the country, Justin determined that he would have to drive about 2,763 miles. What speed would he have to average to complete the trip in no more than 50 hours of driving time?',
            'reasoning': 'Students will perform a compound transformation of a geometric figure within a coordinate plane.'
        },
        3: {
            'description': 'Mathematical Strategic Thinking: Tasks at this level require deep understanding and reasoning, planning, and using evidence.',
            'examples': 'Interpret information from a complex graph, Develop logical arguments for a concept, Solve a multiple-step problem, supported with a mathematical explanation that justifies the answer.',
            'math_example': 'A sweater that you really want has just been placed on sale. The original cost was $63.99. The sale price is $47.99. What is the percent of decrease from the original price? You still do not have enough money saved up to purchase the sweater, so you wait just a little longer and the store now has an ad that states that all items currently on sale have been reduced by 1/3 of the sale price. What is the new sale price?',
            'reasoning': 'Students will perform a geometric transformation to meet specified criteria and then explain what does or does not change about the figure.'
        },
        4: {
            'description': 'Mathematical Extended Thinking: Tasks at this level require complex reasoning, planning, developing, and thinking over an extended period of time.',
            'examples': 'Relate mathematical concepts to real-world applications in new situations, Design a mathematical model to inform and solve a practical or abstract situation.',
            'math_example': 'Students will visit three local grocery stores and find the prices of three different sizes of the same product at the three stores. Students will then determine the unit price for each size item at each store and make a decision as to which is the best buy. Students will then write a report chronicling their work and reporting which is the best buy, justifying their decision with their mathematical work.',
            'reasoning': 'Students will abstract the transformations occurring in an Escher woodprint and then create a simplified tessellation of their own.'
        }
    }
    
    if dok_level in dok_details:
        dok_info = dok_details[dok_level]
        # Construct the full prompt
        prompt = f"Generate a math problem based on the following content: {content}\n" \
                f"Ensure the problem aligns with a Depth of Knowledge level {dok_level}, which requires {dok_info['description']}\n" \
                f"Example of this level: {dok_info['math_example']}\n" \
                f"Reasoning required: {dok_info['reasoning']}"
    else:
        prompt = f"There is no DOK level {dok_level} available. Please select a valid level (1-4)."
    
    return prompt

def get_gpt_response(prompt, client):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Please follow the instructions provided and generate a response."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000  # Adjust tokens as needed to ensure complete responses
          # Add this line to inspect the structure
    )
    try:
        # Attempt to extract the text from the response
        result_text = response.choices[0].message.content.strip()
        return result_text
    except AttributeError as e:
        # Print error message and return a default message if extraction fails
        print("Error extracting text from response:", e)
        return "Failed to generate response due to an unexpected API response format."

if __name__ == "__main__":
    content = input("Enter the educational content/topic: ")
    dok_level = int(input("Enter the Depth of Knowledge (DOK) level (1-4): "))

    prompt = generate_prompt(content, dok_level)
    print("\nGenerated Prompt:")
    print(prompt)

    print("\nAI Generated Response:")
    response = get_gpt_response(prompt)
    print(response)