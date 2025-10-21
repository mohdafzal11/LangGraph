from langgraph.graph import StateGraph , START , END
from typing import TypedDict


class BMIState(TypedDict):
    weight: float 
    height: float 
    bmi:float
    category: str
    
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height ** 2)
    state["bmi"] = bmi
    return state 

def categorize_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"
    state["category"] = category
    return state 
    
graph = StateGraph(BMIState)
graph.add_node("calculate_bmi",calculate_bmi)
graph.add_node("categorize_bmi",categorize_bmi)
graph.add_edge(START,"calculate_bmi")
graph.add_edge("calculate_bmi","categorize_bmi")
graph.add_edge("categorize_bmi",END)


workflow = graph.compile()

initial_state: BMIState = {
    "weight": 80.0,
    "height": 1.73,
}


final_state = workflow.invoke(initial_state)
print(f"Final BMI: {final_state['bmi']}")
print(f"Category: {final_state['category']}")
