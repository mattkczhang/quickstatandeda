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
                max-width: 100%;
                overflow-x: hidden; /* Prevent horizontal scroll */
                font-family: Arial, sans-serif;
                line-height: 1.6;
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
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
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
                font-size: 2.5em;
                font-weight: normal;
                display: inline-block;
            }}

            header nav a {{
                color: white;
                text-decoration: none;
                margin: 0 15px;
                font-size: 1.1em;
                transition: color 0.3s ease;
            }}

            header nav a:hover {{
                color: #ddd;
            }}
            
            /* Dynamic Content Styles */
            .plot {{
                text-align: center;
                margin: 10px 0;
            }}
            .plot img {{
                max-width: 100%; /* Ensures the image is responsive */
                height: auto; /* Maintains aspect ratio */
            }}
            .table-responsive {{
                overflow-x: auto;
            }}
            .table {{
                width: 100%;
                max-width: 100%;
                margin-bottom: 1rem;
                background-color: transparent;
            }}

            /* Subsection Title Styles */
            ol {{
                padding-left: 0;
                list-style: none; /* Remove default list styling */
            }}
            ol li {{
                font-size: 1.2em;
                font-weight: bold;
                padding: 10px 0;
                margin-bottom: 15px;
                background-color: #ffffff;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
                text-align: left;
                padding-left: 20px;
            }}

            /* Button Styles */
            .top-button {{
                margin-top: 20px;
                padding: 10px 15px;
                background-color: #000000; /* Black button */
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}

            .top-button:hover {{
                background-color: #333333;
            }}

            /* Container Styles */
            .container {{
                max-width: 100%;
                margin: auto;
                padding: 20px;
                box-sizing: border-box;
            }}

            /* Section Styles */
            section {{
                padding: 60px 20px;
                text-align: center;
                background-color: #f4f4f9;
                border-bottom: 1px solid #ddd;
                min-height: 150px; /* Add min-height for visibility */
            }}

            section:nth-child(even) {{
                background-color: #e2e2eb;
            }}

            section h2 {{
                font-size: 2em;
                margin-bottom: 10px;
            }}

            /* Footer Styles */
            footer {{
                background-color: #000000; /* Black background */
                color: white;
                text-align: center;
                padding: 15px;
                position: relative;
                bottom: 0;
                width: 100%;
            }}

            /* Responsive Styles */
            @media (max-width: 768px) {{
                header h1 {{
                    font-size: 2em;
                }}

                header nav a {{
                    font-size: 1em;
                    margin: 0 10px;
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
                    font-size: 1.1em;
                    padding: 8px 15px;
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
                <h2>Summary Statistics</h2>
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
            </section>

            <section id="eda">
                <h2>Exploratory Data Analysis</h2>
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
            </section>

            <section id="regression">
                <h2>Preliminary Regression Analysis</h2>
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