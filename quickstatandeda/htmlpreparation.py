def findOutliers(df, col, ):
    """Find outlier records bsed on one column/feature

    Args:
        df (pd.DataFrame): input dataframe
        col (str): column name

    Returns:
        pd.DataFrame: outlier records
    """
    records = df.copy()
    median = records[col].median()    
    deviation_from_med = records[col] - median
    mad = deviation_from_med.abs().median()
    records['modified_z_score'] = deviation_from_med/(0.6745*mad)
    return records[records['modified_z_score'].abs() > 3.5]

def saveInfoToHtml(sum_stats, visuals, regressions, save_path, file_name):
    """Generate an HTML file that showcases the exploratory data analysis
    
    Args:
        sum_stats (dict): dictionary where keys are the section header and values are the summary statistics tables
        visuals (dict): dictionary where keys are the section header and values are the file names of the visuals
        regressions (dict): dictionary where keys are the section header and values are the regression tables
        save_path (str): path to save the HTML file and read the visuals
        file_name (str): file name of the final HTML file
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quick Statistics and Exploratory Data Analysis Report</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-KyZXEAg3QhqLMpG8r+Knujsl5+5hb7O4R0zMQ3f2kZdBc6sP9vO4R0zMQ3f2kZdBc6sP9vO4fVQ8tJaT5fs7iU1z8K6J4t4d1K6Zn6A/FA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
        <style>
            /* Global Styles */
            html, body {{
                margin: 0;
                padding: 0;
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                color: #000000;
                background-color: #ffffff;
                max-width: 100%;
                overflow-x: hidden;
            }}

            /* Header Styles */
            header {{
                background-color: #000000; /* Black background */
                color: white;
                padding: 20px 10px;
                text-align: center;
                position: sticky;
                top: 0;
                width: 100%;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}

            header .logo {{
                display: flex;
                align-items: center;
            }}

            header .logo img {{
                height: 50px;
                margin-right: 10px;
            }}

            header h1 {{
                margin: 0;
                font-size: 2em;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }}

            header nav a {{
                color: white;
                text-decoration: none;
                margin: 0 15px;
                font-size: 1.1em;
                font-weight: 500;
                transition: color 0.3s ease;
            }}

            header nav a:hover {{
                color: #aaaaaa; /* Light gray on hover */
            }}

            /* Dynamic Content Styles */
            .plot {{
                text-align: center;
                margin: 20px 0;
            }}

            .plot img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
            }}

            .table-responsive {{
                overflow-x: auto;
                margin-top: 20px;
            }}

            .table {{
                width: 100%;
                margin-bottom: 20px;
                border-collapse: collapse;
            }}

            .table th, .table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dddddd;
            }}

            .table th {{
                background-color: #000000; /* Black header */
                color: white;
                font-weight: 600;
            }}

            /* Subsection Title Styles */
            ol {{
                padding-left: 0;
                list-style: none;
            }}

            ol li {{
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 20px;
                background-color: #f5f5f5;
                padding: 15px 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
            }}

            /* Button Styles */
            .top-button {{
                display: inline-block;
                margin-top: 30px;
                padding: 12px 20px;
                background-color: #000000; /* Black button */
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                text-transform: uppercase;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }}

            .top-button:hover {{
                background-color: #333333;
                transform: scale(1.05);
            }}

            /* Container Styles */
            .container {{
                max-width: 1100px;
                margin: auto;
                padding: 20px;
            }}

            /* Section Styles */
            section {{
                padding: 60px 20px;
                text-align: center;
                background-color: #ffffff; /* Ensure white background */
                border-bottom: 1px solid #dddddd;
                border-radius: 10px;
                margin-bottom: 40px;
            }}

            section h2 {{
                font-size: 2.2em;
                font-weight: 700;
                color: #000000;
                margin-bottom: 20px;
                text-transform: uppercase;
                letter-spacing: 1.2px;
            }}

            /* Footer Styles */
            footer {{
                background-color: #000000; /* Black background */
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 1em;
                letter-spacing: 1px;
            }}

            /* Collapsible Section Styles */
            details {{
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #ffffff; /* Ensure white background */
            }}

            summary {{
                font-size: 1.5em;
                font-weight: 600;
                cursor: pointer;
                margin-bottom: 10px;
                background-color: #f9f9f9; /* Background color for summary */
                padding: 10px;
                border-radius: 5px;
            }}

            /* Responsive Styles */
            @media (max-width: 768px) {{
                header h1 {{
                    font-size: 1.8em;
                }}

                header nav a {{
                    font-size: 0.9em;
                    margin: 0 8px;
                }}

                header {{
                    flex-direction: column;
                    align-items: center;
                }}

                header nav {{
                    margin-top: 10px;
                }}

                section {{
                    padding: 40px 10px;
                }}

                section h2 {{
                    font-size: 1.8em;
                }}

                ol li {{
                    font-size: 1em;
                    padding: 10px 15px;
                }}

                .top-button {{
                    padding: 10px 15px;
                    font-size: 0.9em;
                }}
            }}
        </style>
    </head>
    <body>

        <!-- Header Section -->
        <header id="top">
            <div class="logo">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQKvslzr0H2trJJ-BvlhfF8WJYu0n1fwrvjrg&s" alt="Logo">
                <h1>Preliminary Study</h1>
            </div>
            <nav>
                <a href="#summary"><i class="fas fa-chart-pie"></i> Summary Statistics</a>
                <a href="#eda"><i class="fas fa-chart-line"></i> Exploratory Data Analysis</a>
                <a href="#regression"><i class="fas fa-chart-bar"></i> Preliminary Regression Analysis</a>
            </nav>
        </header>

        <!-- Main Content -->
        <main class="container">
            <section id="summary">
                <details>
                    <summary>Summary Statistics</summary>
                    <ol>
                    """ 
    for i in sum_stats.keys():
        html_template += f"""
                        <li>{i}
                        <div class="table-responsive">
                            {sum_stats[i].to_html(justify='center', classes='table table-striped', border=2)}
                        </div>
                        </li>
                """
    html_template += """
                    </ol>
                    <a href="#top" class="top-button">Back to Top</a>
                </details>
            </section>

            <section id="eda">
                <details>
                    <summary>Exploratory Data Analysis</summary>
                    <ol>
                    """
    for i in visuals.keys():
        html_template += f"""
                            <li>{i}
                            <div class="plot">
                                <img src="{save_path}visuals/{visuals[i]}" alt="EDA Visual" class="img-fluid">
                            </div>
                            </li>
                        """
    html_template += """
                    </ol>  
                    <a href="#top" class="top-button">Back to Top</a>
                </details>
            </section>

            <section id="regression">
                <details>
                    <summary>Preliminary Regression Analysis</summary>
                    <ol>
                    """
    if len(regressions.keys()) > 0:
        for i in regressions.keys():
            html_template += f"""
                                <li>{i}
                                <div class="table-responsive">
                                    {regressions[i].to_html(index=False, justify='center', classes='table table-striped', border=2)}
                                </div>
                                </li>
                            """
    else:
        html_template += f"""
                            <p> Target feature y is not specified correctly. If the input y is string, check the spelling of the column name.
                            </p>
                                    """
    html_template += """
                    </ol>  
                    <a href="#top" class="top-button">Back to Top</a>         
                </details>
            </section>
        </main>

        <!-- Footer -->
        <footer>
            <p>&copy; Quick Statistics and Exploratory Data Analysis Report</p>
        </footer>

    </body>
    </html>
    """
    
    # Save HTML content to file
    with open(save_path + file_name + '.html', 'w') as file:
        file.write(html_template)