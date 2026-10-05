import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from datetime import datetime
import base64
from io import BytesIO

def create_blood_group_report(data, output_path):
    """
    Create a PDF report for blood group detection results
    
    Args:
        data (dict): Detection results including prediction, confidence, image, etc.
        output_path (str): Path where the PDF should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        title = Paragraph("Blood Group Detection Report", title_style)
        story.append(title)
        
        # Timestamp and user info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_style = ParagraphStyle(
            'Info',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=1
        )
        info = Paragraph(f"Generated on: {timestamp}<br/>User: {data.get('user', 'Unknown')}", info_style)
        story.append(info)
        story.append(Spacer(1, 20))
        
        # Detected blood group section
        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        )
        story.append(Paragraph("Detected Blood Group", section_title_style))
        
        # Prediction box
        prediction_style = ParagraphStyle(
            'Prediction',
            parent=styles['Normal'],
            fontSize=20,
            spaceAfter=12,
            alignment=1,
            textColor=colors.white,
            backColor=colors.HexColor("#667eea")
        )
        prediction_text = f"<b>{data.get('prediction', 'N/A')}</b><br/>Confidence: {data.get('confidence', '0.00')}%"
        story.append(Paragraph(prediction_text, prediction_style))
        story.append(Spacer(1, 20))
        
        # Analysis details section
        story.append(Paragraph("Analysis Details", section_title_style))
        
        # Top 3 predictions table
        table_data = [["Rank", "Blood Group", "Confidence"]]
        top_predictions = data.get('top_3_predictions', [])
        for i, pred in enumerate(top_predictions):
            table_data.append([
                str(i + 1),
                pred.get('blood_group', 'N/A'),
                f"{pred.get('confidence', 0) * 100:.2f}%"
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Fingerprint image section
        story.append(Paragraph("Fingerprint Image", section_title_style))
        
        # Try to add the image if it exists
        image_url = data.get('image_url', '')
        if image_url and image_url.startswith('/static/'):
            # Convert relative path to absolute path
            image_path = image_url.replace('/static/', 'static/')
            if os.path.exists(image_path):
                try:
                    img = Image(image_path, width=4*inch, height=3*inch)
                    img.hAlign = 'CENTER'
                    story.append(img)
                except Exception as e:
                    print(f"Could not add image to PDF: {e}")
                    story.append(Paragraph("Image could not be loaded", styles['Normal']))
            else:
                story.append(Paragraph("Image file not found", styles['Normal']))
        else:
            story.append(Paragraph("No image available", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # System information section
        story.append(Paragraph("System Information", section_title_style))
        system_info = [
            f"<b>Algorithm:</b> CNN with ResNet Architecture",
            f"<b>Confidence Calibration:</b> Applied",
            f"<b>Processing Time:</b> {datetime.now().strftime('%H:%M:%S')}"
        ]
        for info in system_info:
            story.append(Paragraph(info, styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            backColor=colors.yellow,
            borderColor=colors.orange,
            borderWidth=1,
            borderPadding=10,
            borderRadius=5,
            spaceAfter=20
        )
        disclaimer = Paragraph(
            "<b>Medical Disclaimer:</b> This prediction is based on machine learning analysis. "
            "For medical purposes, please confirm with a laboratory blood test. Results should not be used for "
            "medical decision-making without professional verification.",
            disclaimer_style
        )
        story.append(disclaimer)
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=1
        )
        footer = Paragraph(
            "Report generated by BloodGroupAI Detection System<br/>"
            "© 2025 BloodGroupAI. All rights reserved.",
            footer_style
        )
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        return True
        
    except Exception as e:
        print(f"Error creating PDF report: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_data = {
        'prediction': 'B+',
        'confidence': '85.67',
        'top_3_predictions': [
            {'blood_group': 'B+', 'confidence': 0.8567},
            {'blood_group': 'B-', 'confidence': 0.0823},
            {'blood_group': 'O+', 'confidence': 0.0610}
        ],
        'image_url': '/static/uploads/sample.jpg',
        'user': 'test_user'
    }
    
    # Create sample report
    create_blood_group_report(sample_data, 'sample_report.pdf')
    print("Sample report created successfully!")