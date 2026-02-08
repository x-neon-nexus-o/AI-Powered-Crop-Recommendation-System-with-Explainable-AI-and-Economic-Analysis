"""
PDF Report Generator for Crop Recommendation System.
Generates a downloadable PDF with prediction results, economic analysis,
explainability summary, and rotation plan.
"""

import io
import os
from datetime import datetime
from fpdf import FPDF


class CropReportPDF(FPDF):
    """Custom PDF class with header/footer for crop recommendation reports."""

    def __init__(self, crop_name):
        super().__init__()
        self.crop_name = crop_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Crop Recommendation Report', align='L')
        self.cell(0, 8, datetime.now().strftime('%d %b %Y, %I:%M %p'), align='R', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(40, 167, 69)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'AI-Powered Crop Recommendation System | Developed by Prathamesh Bhushan Gawas', align='L')
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='R')

    # ----- Helper methods -----

    def section_title(self, icon_text, title):
        """Draw a colored section header."""
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, f'{icon_text}  {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(40, 167, 69)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def key_value_row(self, key, value, key_width=70):
        """Draw a key-value pair row."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(key_width, 7, key, new_x='END')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), new_x='LMARGIN', new_y='NEXT')

    def badge(self, text, r, g, b):
        """Draw a colored badge/tag."""
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        w = self.get_string_width(text) + 8
        self.cell(w, 7, text, fill=True, align='C', new_x='END')
        self.set_text_color(30, 30, 30)


def generate_crop_report(inputs, predictions, economic_data, rotation_data,
                         explanation_data=None, shap_plot_path=None):
    """
    Generate a complete crop recommendation PDF report.

    Args:
        inputs: dict with N, P, K, temperature, humidity, ph, rainfall, season
        predictions: list of top 3 prediction dicts (crop, probability, confidence, ...)
        economic_data: dict from get_economic_summary() for top crop
        rotation_data: dict from get_rotation_suggestion() for top crop
        explanation_data: optional dict with top_features from SHAP
        shap_plot_path: optional absolute path to SHAP waterfall PNG

    Returns:
        bytes: PDF file content
    """
    top_crop = predictions[0]['crop']
    pdf = CropReportPDF(top_crop)
    pdf.alias_nb_pages()

    # ===== PAGE 1: Title + Inputs + Recommendations =====
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 14, 'Crop Recommendation Report', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, 'AI-Powered Analysis with Explainable AI & Economic Insights',
             align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    # Input Parameters
    pdf.section_title('[INPUT]', 'Your Soil & Climate Parameters')

    param_labels = {
        'N': ('Nitrogen (N)', 'kg/ha'),
        'P': ('Phosphorus (P)', 'kg/ha'),
        'K': ('Potassium (K)', 'kg/ha'),
        'temperature': ('Temperature', '\u00b0C'),
        'humidity': ('Humidity', '%'),
        'ph': ('Soil pH', ''),
        'rainfall': ('Rainfall', 'mm'),
        'season': ('Season', '')
    }

    # Draw input params in a 2-column layout
    col_width = 90
    x_start = 10
    row_count = 0
    for key, (label, unit) in param_labels.items():
        val = inputs.get(key, 'N/A')
        display_val = f'{val} {unit}'.strip() if unit else str(val)

        col = row_count % 2
        if col == 0:
            pdf.set_x(x_start)
        else:
            pdf.set_x(x_start + col_width)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(40, 7, label + ':', new_x='END')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(45, 7, display_val)

        if col == 1:
            pdf.ln(7)
        row_count += 1

    if row_count % 2 == 1:
        pdf.ln(7)
    pdf.ln(4)

    # Top Recommendation
    pdf.section_title('[BEST]', f'Top Recommendation: {top_crop}')

    top = predictions[0]
    prob_pct = top['probability'] * 100

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 12, f'{top_crop}  -  {prob_pct:.1f}% Confidence',
             new_x='LMARGIN', new_y='NEXT')

    # Category badge
    pdf.badge(top.get('category', 'Recommended'), 40, 167, 69)
    pdf.ln(8)

    # Quick economic stats for top crop
    if economic_data:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(60, 60, 60)
        roi = economic_data.get('roi', 0)
        profit = economic_data.get('profit', 0)
        risk = economic_data.get('risk_category', 'N/A')
        pdf.cell(60, 7, f'ROI: {roi:.1f}%', new_x='END')
        pdf.cell(60, 7, f'Profit: Rs.{profit:,.0f}/acre', new_x='END')
        pdf.cell(0, 7, f'Risk: {risk}', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    # Alternative Recommendations
    if len(predictions) > 1:
        pdf.section_title('[ALT]', 'Alternative Recommendations')

        # Table header
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(50, 8, 'Crop', border=1, fill=True, new_x='END')
        pdf.cell(35, 8, 'Confidence', border=1, fill=True, align='C', new_x='END')
        pdf.cell(30, 8, 'ROI', border=1, fill=True, align='C', new_x='END')
        pdf.cell(35, 8, 'Profit (Rs.)', border=1, fill=True, align='C', new_x='END')
        pdf.cell(30, 8, 'Risk', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        for pred in predictions[1:]:
            eco = pred.get('economic', {})
            pdf.cell(50, 8, pred['crop'], border=1, new_x='END')
            pdf.cell(35, 8, pred.get('confidence', 'N/A'), border=1, align='C', new_x='END')
            pdf.cell(30, 8, f"{eco.get('roi', 0):.1f}%", border=1, align='C', new_x='END')
            pdf.cell(35, 8, f"{eco.get('profit', 0):,.0f}", border=1, align='C', new_x='END')
            pdf.cell(30, 8, eco.get('risk_category', 'N/A'), border=1, align='C', new_x='LMARGIN', new_y='NEXT')

        pdf.ln(4)

    # ===== PAGE 2: Economic Analysis =====
    pdf.add_page()
    pdf.section_title('[ECONOMIC]', f'Economic Analysis: {top_crop}')

    if economic_data:
        pdf.key_value_row('Return on Investment (ROI):', f"{economic_data.get('roi', 0):.1f}%")
        pdf.key_value_row('Estimated Profit:', f"Rs.{economic_data.get('profit', 0):,.0f}/acre")
        pdf.key_value_row('Profit Margin:', f"{economic_data.get('profit_margin', 0):.1f}%")
        pdf.key_value_row('Total Revenue:', f"Rs.{economic_data.get('revenue', 0):,.0f}/acre")
        pdf.key_value_row('Total Cost:', f"Rs.{economic_data.get('total_cost', 0):,.0f}/acre")
        pdf.key_value_row('Price Volatility:', f"{economic_data.get('volatility', 0):.1f}%")
        pdf.key_value_row('Risk Category:', economic_data.get('risk_category', 'N/A'))
        pdf.ln(4)

        # Cost Breakdown
        cost_breakdown = economic_data.get('cost_breakdown', {})
        if cost_breakdown:
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 9, 'Cost Breakdown (Rs./acre):', new_x='LMARGIN', new_y='NEXT')

            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_fill_color(240, 240, 240)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(60, 8, 'Category', border=1, fill=True, new_x='END')
            pdf.cell(50, 8, 'Amount (Rs.)', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')

            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(30, 30, 30)
            for category, amount in cost_breakdown.items():
                pdf.cell(60, 8, category.capitalize(), border=1, new_x='END')
                pdf.cell(50, 8, f'{amount:,}', border=1, align='C', new_x='LMARGIN', new_y='NEXT')
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 8, 'Economic data not available for this crop.', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(6)

    # ===== Rotation Plan (same page if space, else new page) =====
    if pdf.get_y() > 180:
        pdf.add_page()

    pdf.section_title('[ROTATION]', f'Crop Rotation Plan: {top_crop}')

    if rotation_data and rotation_data.get('plan'):
        # Rotation plan table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(15, 8, '#', border=1, fill=True, align='C', new_x='END')
        pdf.cell(45, 8, 'Season', border=1, fill=True, new_x='END')
        pdf.cell(55, 8, 'Crop', border=1, fill=True, new_x='END')
        pdf.cell(50, 8, 'Category', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        for i, item in enumerate(rotation_data['plan'], 1):
            is_current = (i == 1)
            if is_current:
                pdf.set_font('Helvetica', 'B', 10)
            else:
                pdf.set_font('Helvetica', '', 10)
            pdf.cell(15, 8, str(i), border=1, align='C', new_x='END')
            pdf.cell(45, 8, item.get('season', ''), border=1, new_x='END')
            pdf.cell(55, 8, item.get('crop', ''), border=1, new_x='END')
            pdf.cell(50, 8, item.get('category', ''), border=1, new_x='LMARGIN', new_y='NEXT')

        pdf.ln(4)

        # Sustainability score
        score = rotation_data.get('sustainability_score', 0)
        rating = rotation_data.get('rating', 'N/A')
        pdf.key_value_row('Sustainability Score:', f'{score}/100 ({rating})')

        benefit = rotation_data.get('benefit', '')
        if benefit:
            pdf.key_value_row('Rotation Benefit:', benefit)

        # Soil impact
        soil_impact = rotation_data.get('soil_impact', {})
        if soil_impact:
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 9, 'Soil Nutrient Impact After Rotation:', new_x='LMARGIN', new_y='NEXT')
            for nutrient, change in soil_impact.items():
                sign = '+' if change >= 0 else ''
                pdf.key_value_row(f'{nutrient}:', f'{sign}{change} kg/ha')
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 8, 'Rotation plan not available for this crop.', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(6)

    # ===== PAGE 3: AI Explanation (if available) =====
    if explanation_data and explanation_data.get('top_features'):
        pdf.add_page()
        pdf.section_title('[XAI]', f'AI Explanation: Why {top_crop}?')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 6,
                        'The following features were most influential in the AI model\'s '
                        f'decision to recommend {top_crop}. Positive values support the '
                        'recommendation; negative values work against it.')
        pdf.ln(4)

        # Feature contribution table
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(10, 8, '#', border=1, fill=True, align='C', new_x='END')
        pdf.cell(55, 8, 'Feature', border=1, fill=True, new_x='END')
        pdf.cell(35, 8, 'Your Value', border=1, fill=True, align='C', new_x='END')
        pdf.cell(40, 8, 'SHAP Impact', border=1, fill=True, align='C', new_x='END')
        pdf.cell(40, 8, 'Direction', border=1, fill=True, align='C', new_x='LMARGIN', new_y='NEXT')

        feature_labels = {
            'N': 'Nitrogen (N)',
            'P': 'Phosphorus (P)',
            'K': 'Potassium (K)',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'ph': 'Soil pH',
            'rainfall': 'Rainfall'
        }

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        for i, feat in enumerate(explanation_data['top_features'], 1):
            name = feat.get('feature', '')
            label = feature_labels.get(name, name)
            value = inputs.get(name, feat.get('value', 'N/A'))
            shap_val = feat.get('shap_value', 0)
            direction = 'Supports' if shap_val > 0 else 'Against'

            # Color the direction
            pdf.cell(10, 8, str(i), border=1, align='C', new_x='END')
            pdf.cell(55, 8, label, border=1, new_x='END')
            pdf.cell(35, 8, str(value), border=1, align='C', new_x='END')
            pdf.cell(40, 8, f'{shap_val:+.3f}', border=1, align='C', new_x='END')

            if shap_val > 0:
                pdf.set_text_color(40, 167, 69)
            else:
                pdf.set_text_color(220, 53, 69)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(40, 8, direction, border=1, align='C', new_x='LMARGIN', new_y='NEXT')
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(30, 30, 30)

        pdf.ln(4)

        # Summary line
        positive_count = sum(1 for f in explanation_data['top_features'] if f.get('shap_value', 0) > 0)
        total = len(explanation_data['top_features'])
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(40, 167, 69)
        pdf.cell(0, 8,
                 f'{positive_count} of {total} key factors support {top_crop} as the best choice.',
                 new_x='LMARGIN', new_y='NEXT')

        # Embed SHAP waterfall plot if available
        if shap_plot_path and os.path.isfile(shap_plot_path):
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 9, 'SHAP Waterfall Plot:', new_x='LMARGIN', new_y='NEXT')

            # Check if we need a new page for the image
            if pdf.get_y() > 160:
                pdf.add_page()

            try:
                pdf.image(shap_plot_path, x=15, w=180)
            except Exception:
                pdf.set_font('Helvetica', 'I', 9)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 8, '(SHAP plot image could not be embedded)',
                         new_x='LMARGIN', new_y='NEXT')

    # Output to bytes
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)
