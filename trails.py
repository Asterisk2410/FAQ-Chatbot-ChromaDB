# import pandas as pd
 
# # Define the data for the workflow pipeline
# data = {
#     "Phase": [
#         "Prototyping", "Prototyping", "Prototyping", 
#         "Production", "Production", "Production", "Production", "Production", "Production"
#     ],
#     "Step": [
#         "Basic Mechanics Development", "Initial Feedback and Iteration", "Prototype Enhancement", 
#         "Detailed Art and Design", "Advanced Programming", "Sound and Music Integration", 
#         "Level Design and Balancing", "Alpha Testing and Iteration", "Beta Testing and Final Adjustments"
#     ],
#     "Duration (Weeks)": [
#         2-3, 1-2, 2-3, 
#         4-6, 4-6, 3-4, 
#         4-5, 3-4, 4-5
#     ],
#     "Tasks": [
#         "Implement basic character movement, Implement gravity flip mechanics, Create basic level layout, Set up basic collision detection and physics",
#         "Gather feedback, Identify and fix major issues, Refine mechanics based on feedback",
#         "Add basic animations, Implement basic UI elements, Introduce simple obstacles and collectibles, Create 2-3 prototype levels",
#         "Finalize concept art, Create detailed character models and animations, Design and create environment assets, Design and create obstacles and collectibles",
#         "Implement advanced mechanics, Develop detailed level layouts, Create and integrate additional UI elements, Implement saving and loading functionality",
#         "Create and integrate sound effects, Compose background music, Record and integrate voiceovers",
#         "Design and develop all game levels, Ensure levels are balanced, Playtest levels to fix pacing or difficulty spikes",
#         "Conduct internal alpha testing, Gather feedback, Iterate on design and mechanics",
#         "Conduct external beta testing, Collect and analyze feedback, Make final adjustments, Optimize game for performance"
#     ]
# }
 
# # Create a DataFrame
# df = pd.DataFrame(data)
 
# # Save to an Excel file
# file_path = r'data/Gravity_Flip_Workflow_Pipeline.xlsx'
# df.to_excel(file_path, index=False)
 
# file_path

# import openpyxl
# from openpyxl.styles import PatternFill
 
# # Create a new workbook and select the active sheet
# wb = openpyxl.Workbook()
# ws = wb.active
 
# # Define headers
# headers = ["Phase", "Step", "Duration (Weeks)", "Tasks"]
 
# # Add headers to the sheet
# for col_num, header in enumerate(headers, 1):
#     ws.cell(row=1, column=col_num, value=header)
 
# # Define the data for the workflow pipeline
# data = [
#     ["Prototyping", "Basic Mechanics Development", "2-3", 
#      "Implement basic character movement, Implement gravity flip mechanics, Create basic level layout, Set up basic collision detection and physics"],
#     ["Prototyping", "Initial Feedback and Iteration", "1-2", 
#      "Gather feedback, Identify and fix major issues, Refine mechanics based on feedback"],
#     ["Prototyping", "Prototype Enhancement", "2-3", 
#      "Add basic animations, Implement basic UI elements, Introduce simple obstacles and collectibles, Create 2-3 prototype levels"],
#     ["Production", "Detailed Art and Design", "4-6", 
#      "Finalize concept art, Create detailed character models and animations, Design and create environment assets, Design and create obstacles and collectibles"],
#     ["Production", "Advanced Programming", "4-6", 
#      "Implement advanced mechanics, Develop detailed level layouts, Create and integrate additional UI elements, Implement saving and loading functionality"],
#     ["Production", "Sound and Music Integration", "3-4", 
#      "Create and integrate sound effects, Compose background music, Record and integrate voiceovers"],
#     ["Production", "Level Design and Balancing", "4-5", 
#      "Design and develop all game levels, Ensure levels are balanced, Playtest levels to fix pacing or difficulty spikes"],
#     ["Production", "Alpha Testing and Iteration", "3-4", 
#      "Conduct internal alpha testing, Gather feedback, Iterate on design and mechanics"],
#     ["Production", "Beta Testing and Final Adjustments", "4-5", 
#      "Conduct external beta testing, Collect and analyze feedback, Make final adjustments, Optimize game for performance"]
# ]
 
# # Add data to the sheet
# for row_num, row_data in enumerate(data, 2):
#     for col_num, cell_value in enumerate(row_data, 1):
#         ws.cell(row=row_num, column=col_num, value=cell_value)
 
# # Define color fills
# prototyping_fill = PatternFill(start_color="FFFF99", end_color="FFFF99", fill_type="solid")  # Light Yellow
# production_fill = PatternFill(start_color="99CCFF", end_color="99CCFF", fill_type="solid")  # Light Blue
 
# # Apply color fills
# for row_num, row_data in enumerate(data, 2):
#     phase = row_data[0]
#     for col_num in range(1, len(headers) + 1):
#         if phase == "Prototyping":
#             ws.cell(row=row_num, column=col_num).fill = prototyping_fill
#         elif phase == "Production":
#             ws.cell(row=row_num, column=col_num).fill = production_fill
 
# # Save the workbook
# file_path_detailed = r'data/Gravity_Flip_Workflow_Pipeline_Detailed.xlsx'
# wb.save(file_path_detailed)
 
# file_path_detailed

import sys , time, torch, numpy

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import llamacpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"torch version: {torch.__version__}")
    
if __name__ == "__main__":
    main()