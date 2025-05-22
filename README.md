# Document Search Chatbot

An interactive chatbot that performs research across multiple documents, identifies common themes, and provides detailed, cited responses to user queries.

## Features

- Document Upload & Processing
  - Support for PDF and scanned image documents
  - OCR processing for scanned documents
  - Text extraction and preprocessing
  - Document storage and management

- Advanced Query Processing
  - Natural language query support
  - Individual document processing
  - Precise citation extraction
  - Theme identification across documents

- User Interface
  - Modern web interface with Tailwind CSS and Bootstrap
  - Document management dashboard
  - Interactive chat interface
  - Citation visualization

## Technical Stack

- Backend: Flask (Python)
- Frontend: HTML, Tailwind CSS, Bootstrap
- AI/ML: Google Gemini Pro, LangChain
- Database: ChromaDB (Vector Database)
- OCR: Tesseract
- Document Processing: pdf2image, Pillow

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd doc-research-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

5. Run the application:
```bash
python run.py
```

## Project Structure

```
doc-research-bot/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core functionality
│   ├── models/        # Data models
│   ├── services/      # Business logic
│   ├── static/        # Static files
│   ├── templates/     # HTML templates
│   ├── main.py        # Application entry point
│   └── config.py      # Configuration
├── data/              # Document storage
├── tests/             # Test files
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload documents through the interface
3. Wait for document processing
4. Start asking questions in the chat interface
5. View responses with citations and theme analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
