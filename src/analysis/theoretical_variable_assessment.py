#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theoretical Variable Assessment

This script creates a theoretical framework for variable selection based on
identified predictors from the EDA synthesis and literature review. It documents
expected relationships and generates a comprehensive variable assessment report.

Author: Daniel Tierney
Date: April 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output" / "variable_assessment"
REPORTS_DIR = BASE_DIR / "reports"
EDA_REPORT = REPORTS_DIR / "eda_synthesis_and_planning.md"
LIT_REVIEW = BASE_DIR / "RP2_ Lit Review Annotations.md"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_section(section_name):
    """Log a section header to improve log readability."""
    logger.info(f"\n{'=' * 40}\n{section_name}\n{'=' * 40}")

def load_eda_synthesis():
    """Load the EDA synthesis report to extract key findings and predictors."""
    log_section("Loading EDA Synthesis")
    
    if not EDA_REPORT.exists():
        logger.error(f"EDA synthesis report not found at {EDA_REPORT}")
        return None
    
    with open(EDA_REPORT, 'r') as f:
        eda_content = f.read()
        
    logger.info(f"Loaded EDA synthesis report ({len(eda_content)} characters)")
    
    # Extract strongest predictors section
    strongest_predictors = {}
    if "## 3. Strongest Demographic Predictors of Performance" in eda_content:
        predictors_section = eda_content.split("## 3. Strongest Demographic Predictors of Performance")[1].split("##")[0]
        
        # Parse the table of predictors
        table_lines = [line.strip() for line in predictors_section.split("\n") if "|" in line]
        if len(table_lines) > 2:  # Header + separator + at least one row
            for line in table_lines[2:]:  # Skip header and separator
                cells = [cell.strip() for cell in line.split("|")[1:-1]]  # Skip first and last empty cells
                if len(cells) >= 3:
                    var_name, score, justification = cells
                    try:
                        strongest_predictors[var_name] = {
                            "score": int(score),
                            "justification": justification
                        }
                    except ValueError:
                        # Skip if score isn't a number
                        pass
    
    logger.info(f"Extracted {len(strongest_predictors)} predictors from EDA synthesis")
    for var, details in sorted(strongest_predictors.items(), key=lambda x: x[1]['score'], reverse=True)[:5]:
        logger.info(f"  - {var}: Score {details['score']}, Justification: {details['justification']}")
    
    return {
        "content": eda_content,
        "strongest_predictors": strongest_predictors
    }

def load_literature_review():
    """Load and parse the literature review to extract relevant insights."""
    log_section("Loading Literature Review")
    
    if not LIT_REVIEW.exists():
        logger.error(f"Literature review not found at {LIT_REVIEW}")
        return None
    
    with open(LIT_REVIEW, 'r') as f:
        lit_content = f.read()
    
    logger.info(f"Loaded literature review ({len(lit_content)} characters)")
    
    # Extract socio-economic determinants section
    socioeconomic_section = ""
    if "# Socio-Economic Determinants" in lit_content:
        socioeconomic_section = lit_content.split("# Socio-Economic Determinants")[1].split("#")[0].strip()
    
    # Extract key variables mentioned in literature review
    key_variables = {
        "education": {
            "mentions": lit_content.lower().count("education"),
            "variants": ["third level", "basic education", "secondary education", "education rate", "education level"],
            "sources": []
        },
        "employment": {
            "mentions": lit_content.lower().count("employment"),
            "variants": ["employment rate", "unemployment", "labor force", "economic activity"],
            "sources": []
        },
        "social_class": {
            "mentions": lit_content.lower().count("social class") + lit_content.lower().count("socioeconomic status"),
            "variants": ["professional", "working class", "socioeconomic", "income", "social class"],
            "sources": []
        },
        "population": {
            "mentions": lit_content.lower().count("population"),
            "variants": ["youth", "demographics", "density", "age", "gender ratio"],
            "sources": []
        },
        "environmental": {
            "mentions": lit_content.lower().count("environmental") + lit_content.lower().count("geography"),
            "variants": ["elevation", "rainfall", "climate", "physical geography"],
            "sources": []
        },
        "urban_rural": {
            "mentions": lit_content.lower().count("urban") + lit_content.lower().count("rural"),
            "variants": ["urban", "rural", "community size", "settlement"],
            "sources": []
        }
    }
    
    # Simplified method to extract sources - use regular expressions
    import re
    
    # Find source titles and authors
    source_titles = re.findall(r'Source Title:\s*\n(.*?)\n', lit_content)
    source_authors = re.findall(r'Authors:\s*\n(.*?)\n', lit_content)
    
    # Create list of source info
    sources = []
    for i in range(min(len(source_titles), len(source_authors))):
        sources.append({
            "title": source_titles[i].strip(),
            "author": source_authors[i].strip()
        })
    
    # Assign sources to categories based on keyword matching
    for category in key_variables:
        for source in sources:
            # Create a search string that combines title and author
            search_text = f"{source['title']} {source['author']}".lower()
            
            # Check if category or variants are mentioned
            if category.lower() in search_text or any(variant.lower() in search_text for variant in key_variables[category]["variants"]):
                key_variables[category]["sources"].append(source['title'])
    
    logger.info("Extracted key variables from literature review:")
    for var, details in sorted(key_variables.items(), key=lambda x: x[1]['mentions'], reverse=True):
        logger.info(f"  - {var}: {details['mentions']} mentions, {len(details['sources'])} sources")
    
    return {
        "content": lit_content,
        "socioeconomic_section": socioeconomic_section,
        "key_variables": key_variables
    }

def create_variable_hypotheses(eda_synthesis, lit_review):
    """Create variable importance hypotheses based on identified predictors and literature review."""
    log_section("Creating Variable Importance Hypotheses")
    
    # Combine top predictors from EDA with literature support
    hypotheses = {}
    
    # Begin with EDA-identified predictors
    for var, details in sorted(eda_synthesis["strongest_predictors"].items(), key=lambda x: x[1]['score'], reverse=True):
        hypotheses[var] = {
            "hypothesis": "",
            "evidence": {
                "eda_score": details["score"],
                "eda_justification": details["justification"],
                "lit_support": 0,
                "lit_sources": []
            }
        }
    
    # Check for literature support for EDA predictors
    for var in hypotheses:
        var_lower = var.lower()
        for lit_var, lit_details in lit_review["key_variables"].items():
            if var_lower in lit_var.lower() or any(variant.lower() in var_lower for variant in lit_details["variants"]):
                hypotheses[var]["evidence"]["lit_support"] = lit_details["mentions"]
                hypotheses[var]["evidence"]["lit_sources"] = lit_details["sources"]
    
    # Generate specific hypotheses for each variable based on evidence
    hypotheses_text = {
        "Elevation": "Areas with higher elevation are associated with stronger GAA club performance, particularly in rural settings.",
        "basic_education_rate": "Areas with higher basic education rates show poorer club performance outcomes, reflecting socioeconomic challenges.",
        "working_class_rate": "Areas with higher working class populations tend to have lower club performance levels, due to resource constraints.",
        "urban": "Urban clubs generally outperform rural clubs due to larger population base and better access to facilities.",
        "third_level_rate": "Areas with higher third-level education rates show stronger club performance outcomes due to socioeconomic advantages.",
        "employment_rate": "Higher employment rates in a club's catchment area positively correlate with club performance.",
        "youth_proportion": "Areas with higher youth proportions support better long-term club sustainability and performance.",
        "professional_rate": "Higher proportions of professionals in a catchment area correlate with stronger club performance.",
        "population_density": "Higher population density supports better club performance through larger talent pools."
    }
    
    # Apply specific hypothesis text where available, generate generic ones otherwise
    for var in hypotheses:
        if var in hypotheses_text:
            hypotheses[var]["hypothesis"] = hypotheses_text[var]
        else:
            # Generic hypothesis based on variable name
            if "rate" in var or "proportion" in var:
                hypotheses[var]["hypothesis"] = f"Higher {var} is associated with improved club performance outcomes."
            else:
                hypotheses[var]["hypothesis"] = f"{var.capitalize()} is a significant predictor of club performance outcomes."
    
    logger.info("Generated variable importance hypotheses:")
    for var, details in sorted(hypotheses.items(), key=lambda x: x[1]['evidence']['eda_score'], reverse=True)[:5]:
        logger.info(f"  - {var}: {details['hypothesis']}")
        logger.info(f"    Evidence: EDA score {details['evidence']['eda_score']}, Literature support: {details['evidence']['lit_support']} mentions")
    
    return hypotheses

def document_expected_relationships(hypotheses, eda_synthesis, lit_review):
    """Document expected relationships with focus on education, employment, and social class variables."""
    log_section("Documenting Expected Relationships")
    
    relationships = {
        "education": {
            "variables": ["third_level_rate", "basic_education_rate", "secondary_education_rate"],
            "relationship": "Higher third-level education rates generally correlate with better club performance, while higher basic education rates (as highest level attained) tend to correlate with poorer performance outcomes. This reflects the socioeconomic advantage associated with higher education levels.",
            "mechanism": "Education levels influence club performance through several mechanisms: 1) resource availability for club support, 2) social capital and networking opportunities, 3) greater awareness of sports benefits, and 4) higher likelihood of community investment in club activities.",
            "literature_evidence": []
        },
        "employment": {
            "variables": ["employment_rate", "unemployment_rate", "labor_force_participation_rate"],
            "relationship": "Higher employment rates and labor force participation rates positively correlate with club success metrics. Areas with higher unemployment tend to have weaker club performance due to resource constraints.",
            "mechanism": "Employment affects club performance through: 1) greater financial resources for club participation, 2) structured time management that supports volunteering, 3) broader social networks that strengthen club organization, and 4) stronger community economic base that enables better facilities.",
            "literature_evidence": []
        },
        "social_class": {
            "variables": ["professional_rate", "working_class_rate", "socioeconomic_index"],
            "relationship": "Higher proportions of professionals and higher socioeconomic status correlate with better club performance, while higher working-class rates typically relate to lower performance metrics.",
            "mechanism": "Social class impacts club performance via: 1) differential resource availability for participation, 2) varying levels of social capital that affect club organization, 3) different time constraints affecting volunteering, and 4) historical patterns of sport participation across class groups.",
            "literature_evidence": []
        },
        "demographic": {
            "variables": ["youth_proportion", "population_density", "school_age_rate"],
            "relationship": "Higher youth proportions and larger overall population densities generally support better club performance through larger talent pools and greater community engagement.",
            "mechanism": "Demographic factors affect performance through: 1) larger potential player base, 2) enhanced competition within the club improving standards, 3) greater supporter base generating resources, and 4) broader volunteer pool strengthening club administration.",
            "literature_evidence": []
        },
        "environmental": {
            "variables": ["Elevation", "annual_rainfall", "rain_days"],
            "relationship": "Environmental factors like elevation show an unexpected but consistent relationship with club performance. Areas at higher elevations tend to have stronger club performance.",
            "mechanism": "Environmental factors may influence performance through: 1) physical conditioning benefits from training at elevation, 2) historical patterns of settlement and community formation, 3) correlation with specific socioeconomic patterns, and 4) impact on facility usage and availability.",
            "literature_evidence": []
        },
        "urban_rural": {
            "variables": ["urban", "rural", "urban_access"],
            "relationship": "Urban clubs generally outperform rural clubs, particularly in football, though this relationship is complex and affected by local factors.",
            "mechanism": "Urban/rural status affects performance through: 1) population density and talent pool size, 2) access to facilities and resources, 3) competition proximity and intensity, and 4) historical patterns of sport development.",
            "literature_evidence": []
        }
    }
    
    # Extract literature evidence for each category
    for category in relationships:
        # Find related literature sources
        for var, details in lit_review["key_variables"].items():
            if var in category or category in var:
                relationships[category]["literature_evidence"] = details["sources"][:3]  # Top 3 sources
    
    # Extract expected direction and strength for top variables
    directional_relationships = {}
    key_vars = ["third_level_rate", "basic_education_rate", "working_class_rate", 
                "employment_rate", "youth_proportion", "Elevation", "urban", "professional_rate"]
    
    for var in key_vars:
        if var in hypotheses:
            direction = "positive" if "positive" in hypotheses[var]["hypothesis"].lower() or "higher" in hypotheses[var]["hypothesis"].lower() else "negative"
            strength = "strong" if hypotheses[var]["evidence"]["eda_score"] > 15 else "moderate" if hypotheses[var]["evidence"]["eda_score"] > 8 else "weak"
            
            directional_relationships[var] = {
                "direction": direction,
                "strength": strength,
                "explanation": hypotheses[var]["hypothesis"]
            }
    
    logger.info("Documented expected relationships:")
    for category, details in relationships.items():
        logger.info(f"  - {category.capitalize()} variables: {', '.join(details['variables'][:3])}")
        logger.info(f"    Evidence sources: {len(details['literature_evidence'])} sources")
    
    logger.info("Specific variable relationships:")
    for var, details in directional_relationships.items():
        logger.info(f"  - {var}: {details['strength']} {details['direction']} relationship")
    
    return {
        "category_relationships": relationships,
        "directional_relationships": directional_relationships
    }

def generate_assessment_report(hypotheses, relationships, eda_synthesis, lit_review):
    """Generate a comprehensive variable assessment report."""
    log_section("Generating Variable Assessment Report")
    
    report_path = REPORTS_DIR / "theoretical_variable_assessment.md"
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Theoretical Variable Assessment Report\n\n")
        f.write(f"*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Introduction
        f.write("## 1. Introduction\n\n")
        f.write("This report presents a theoretical assessment of key variables identified through exploratory data analysis (EDA) ")
        f.write("and supported by literature review. It documents hypothesized relationships between demographic variables and club performance, ")
        f.write("providing a foundation for variable selection in the statistical modeling phase.\n\n")
        
        # Theoretical Framework
        f.write("## 2. Theoretical Framework\n\n")
        f.write("The assessment is grounded in a socio-ecological framework that recognizes multiple layers of influence on GAA club success. ")
        f.write("These include:\n\n")
        
        f.write("1. **Demographic environment**: Population size, youth proportion, gender distribution\n")
        f.write("2. **Socioeconomic conditions**: Education levels, employment patterns, social class distribution\n")
        f.write("3. **Physical environment**: Geographic features, elevation, rainfall patterns\n")
        f.write("4. **Cultural context**: Urban/rural setting, historical sport engagement\n")
        f.write("5. **Resource availability**: Infrastructure, volunteer capacity, financial support\n\n")
        
        f.write("This framework recognizes that club performance is shaped by complex interactions between these layers, ")
        f.write("with certain factors having both direct and indirect effects on club outcomes.\n\n")
        
        # Literature Support
        f.write("## 3. Literature Evidence\n\n")
        f.write("The literature on socio-economic determinants of sports performance provides strong support for several key relationships:\n\n")
        
        # Add literature evidence by category
        for category, details in relationships["category_relationships"].items():
            f.write(f"### 3.{list(relationships['category_relationships'].keys()).index(category) + 1} {category.capitalize()} Factors\n\n")
            f.write(f"{details['relationship']}\n\n")
            
            if details["literature_evidence"]:
                f.write("**Supporting Literature:**\n\n")
                for source in details["literature_evidence"]:
                    f.write(f"- {source}\n")
                f.write("\n**Proposed Mechanism:**\n\n")
                f.write(f"{details['mechanism']}\n\n")
            else:
                f.write("**Limited literature directly addresses this relationship in GAA context.**\n\n")
        
        # Variable Importance Hypotheses
        f.write("## 4. Variable Importance Hypotheses\n\n")
        f.write("Based on both EDA findings and literature review, the following hypotheses have been developed ")
        f.write("regarding the relationship between key predictors and club performance:\n\n")
        
        for var, details in sorted(hypotheses.items(), key=lambda x: x[1]['evidence']['eda_score'], reverse=True)[:10]:
            f.write(f"### 4.{list(sorted(hypotheses.items(), key=lambda x: x[1]['evidence']['eda_score'], reverse=True)[:10]).index((var, details)) + 1} {var}\n\n")
            f.write(f"**Hypothesis:** {details['hypothesis']}\n\n")
            f.write(f"**Evidence Strength:** {'Strong' if details['evidence']['eda_score'] > 15 else 'Moderate' if details['evidence']['eda_score'] > 8 else 'Weak'}\n\n")
            f.write(f"**EDA Support:** Score {details['evidence']['eda_score']}, \"{details['evidence']['eda_justification']}\"\n\n")
            
            if details['evidence']['lit_sources']:
                f.write("**Literature Support:**\n\n")
                for source in details['evidence']['lit_sources'][:3]:
                    f.write(f"- {source}\n")
                f.write("\n")
            else:
                f.write("**Literature Support:** Limited direct evidence in reviewed literature\n\n")
        
        # Expected Relationships
        f.write("## 5. Expected Relationships\n\n")
        f.write("This section details the expected relationships between key variables and club performance metrics, ")
        f.write("with particular focus on education and social class variables which show the strongest evidence base.\n\n")
        
        f.write("### 5.1 Education Variables\n\n")
        
        f.write("| Variable | Direction | Strength | Explanation |\n")
        f.write("|----------|-----------|----------|-------------|\n")
        for var in ["third_level_rate", "basic_education_rate", "secondary_education_rate"]:
            if var in relationships["directional_relationships"]:
                details = relationships["directional_relationships"][var]
                f.write(f"| {var} | {details['direction'].capitalize()} | {details['strength'].capitalize()} | {details['explanation']} |\n")
        f.write("\n")
        
        f.write("### 5.2 Social Class Variables\n\n")
        
        f.write("| Variable | Direction | Strength | Explanation |\n")
        f.write("|----------|-----------|----------|-------------|\n")
        for var in ["professional_rate", "working_class_rate"]:
            if var in relationships["directional_relationships"]:
                details = relationships["directional_relationships"][var]
                f.write(f"| {var} | {details['direction'].capitalize()} | {details['strength'].capitalize()} | {details['explanation']} |\n")
        f.write("\n")
        
        f.write("### 5.3 Other Key Variables\n\n")
        
        f.write("| Variable | Direction | Strength | Explanation |\n")
        f.write("|----------|-----------|----------|-------------|\n")
        for var in ["employment_rate", "youth_proportion", "Elevation", "urban"]:
            if var in relationships["directional_relationships"]:
                details = relationships["directional_relationships"][var]
                f.write(f"| {var} | {details['direction'].capitalize()} | {details['strength'].capitalize()} | {details['explanation']} |\n")
        f.write("\n")
        
        # Implications for Variable Selection
        f.write("## 6. Implications for Variable Selection\n\n")
        f.write("Based on the theoretical assessment, the following recommendations are made for variable selection:\n\n")
        
        f.write("### 6.1 Primary Variables\n\n")
        
        f.write("The following variables should be prioritized in the modeling phase due to their strong theoretical ")
        f.write("and empirical support:\n\n")
        
        # List top variables from EDA with literature support
        top_vars = [var for var, details in sorted(hypotheses.items(), 
                                                 key=lambda x: (x[1]['evidence']['eda_score'] + x[1]['evidence']['lit_support']), 
                                                 reverse=True)[:5]]
        for var in top_vars:
            f.write(f"- **{var}**: Combined evidence score {hypotheses[var]['evidence']['eda_score'] + hypotheses[var]['evidence']['lit_support']}\n")
        f.write("\n")
        
        f.write("### 6.2 Variable Transformations\n\n")
        
        f.write("The following transformations should be considered:\n\n")
        
        f.write("- **Standardization**: For all variables to facilitate comparison of effect sizes\n")
        f.write("- **Log transformation**: For highly skewed variables, particularly environmental variables (e.g., Elevation)\n")
        f.write("- **Interaction terms**: Specifically between:\n")
        f.write("  - Education variables and urban/rural status\n")
        f.write("  - Employment and social class variables\n")
        f.write("  - Environmental factors and demographic variables\n\n")
        
        f.write("### 6.3 Multicollinearity Considerations\n\n")
        
        f.write("Particular attention should be paid to potential multicollinearity among:\n\n")
        
        f.write("- Education variables (`third_level_rate`, `basic_education_rate`, `secondary_education_rate`)\n")
        f.write("- Employment and social class variables (`employment_rate`, `professional_rate`, `working_class_rate`)\n")
        f.write("- Demographic and urban/rural variables (`population_density`, `urban`)\n\n")
        
        f.write("Principal Component Analysis (PCA) should be implemented to create composite variables for highly correlated predictors.\n\n")
        
        # Conclusion
        f.write("## 7. Conclusion\n\n")
        f.write("This theoretical variable assessment has established a solid foundation for the variable selection phase. ")
        f.write("By integrating insights from exploratory data analysis with evidence from literature review, ")
        f.write("it has identified key predictor variables and hypothesized relationships with club performance metrics. ")
        f.write("The assessment supports a focus on education, social class, and employment variables, ")
        f.write("with consideration of environmental and demographic factors as important moderators. ")
        f.write("These insights will guide the implementation of feature selection techniques and model development ")
        f.write("in the next phase of the research.\n")
    
    logger.info(f"Generated variable assessment report at {report_path}")
    
    # Also generate a visualization of variable importance scores
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for visualization
    var_names = []
    eda_scores = []
    lit_scores = []
    
    for var, details in sorted(hypotheses.items(), key=lambda x: x[1]['evidence']['eda_score'] + x[1]['evidence']['lit_support'], reverse=True)[:10]:
        var_names.append(var)
        eda_scores.append(details['evidence']['eda_score'])
        lit_scores.append(details['evidence']['lit_support'])
    
    # Reverse lists for bottom-up plotting
    var_names.reverse()
    eda_scores.reverse()
    lit_scores.reverse()
    
    width = 0.35
    y_pos = np.arange(len(var_names))
    
    ax.barh(y_pos, eda_scores, width, label='EDA Evidence', color='#1f77b4')
    ax.barh(y_pos + width, lit_scores, width, label='Literature Support', color='#ff7f0e')
    
    ax.set_yticks(y_pos + width/2)
    ax.set_yticklabels(var_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Evidence Score')
    ax.set_title('Variable Importance: Combined Evidence from EDA and Literature')
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    viz_path = OUTPUT_DIR / "variable_importance_scores.png"
    plt.savefig(viz_path)
    plt.close()
    
    logger.info(f"Generated variable importance visualization at {viz_path}")
    
    return report_path

def main():
    """Main function to execute the theoretical variable assessment workflow."""
    log_section("Starting Theoretical Variable Assessment")
    
    try:
        # Load required data
        eda_synthesis = load_eda_synthesis()
        lit_review = load_literature_review()
        
        if not eda_synthesis or not lit_review:
            logger.error("Failed to load required data. Exiting.")
            return
        
        # Create variable hypotheses
        hypotheses = create_variable_hypotheses(eda_synthesis, lit_review)
        
        # Document expected relationships
        relationships = document_expected_relationships(hypotheses, eda_synthesis, lit_review)
        
        # Generate assessment report
        report_path = generate_assessment_report(hypotheses, relationships, eda_synthesis, lit_review)
        
        logger.info(f"Theoretical Variable Assessment completed successfully")
        logger.info(f"Report available at: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in Theoretical Variable Assessment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 