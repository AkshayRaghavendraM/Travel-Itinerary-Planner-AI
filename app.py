from flask import Flask, render_template, request, send_file
from fpdf import FPDF
import tempfile
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

app = Flask(__name__)

# Define PlannerState for itinerary
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# Define the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)  # Default font (Helvetica, Arial, Times)
        self.cell(0, 10, 'Day Trip Itinerary', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_itinerary(self, text):
        self.set_font('Arial', '', 12)
        # Replace ₹ with 'Rs.' (as Arial doesn't support ₹)
        text = text.replace("₹", "Rs.")
        self.multi_cell(0, 10, text)

def generate_pdf(itinerary_text):
    pdf = PDF()
    pdf.add_page()
    pdf.add_itinerary(itinerary_text)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    return temp_file.name

def travel_planner(city: str, interests: str):
    state = {"messages": [], "city": "", "interests": [], "itinerary": ""}
    
    # Process inputs
    state['city'] = city
    state['interests'] = [interest.strip() for interest in interests.split(',')]
    
    # Generate itinerary
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state['city'], 
        interests=", ".join(state['interests'])
    ))
    itinerary = response.content
    
    # Generate PDF
    pdf_path = generate_pdf(itinerary)
    return itinerary, pdf_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        interests = request.form['interests']
        itinerary, pdf_path = travel_planner(city, interests)
        return render_template('index.html', 
                             itinerary=itinerary,
                             pdf_path=pdf_path,
                             show_results=True)
    return render_template('index.html', show_results=False)

@app.route('/download')
def download():
    pdf_path = request.args.get('pdf_path')
    return send_file(pdf_path, as_attachment=True, download_name='itinerary.pdf')

if __name__ == '__main__':
    app.run(debug=True)
