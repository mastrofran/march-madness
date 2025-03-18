# first line: 89
@memory.cache
def scrape_team_rankings(year, stat_url):
    """Scrape team statistics using pandas read_html with fuzzy cleaning"""
    url = f"https://www.teamrankings.com/ncaa-basketball/{stat_url}?date={year}-03-18"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Use StringIO to avoid pandas warning
        html = BeautifulSoup(response.content, 'html.parser')
        html_io = StringIO(str(html))
        
        tables = pd.read_html(html_io)
        if not tables:
            return pd.DataFrame()
            
        table = tables[0].iloc[:, 1:3]
        table.columns = ['Team', 'Value']
        
        # Clean team names
        table['Team'] = (
            table['Team']
            .astype(str)
            .str.replace(r'\s*\(\d+\)', '', regex=True)  # Remove rankings
            .str.replace(r'\s*\(\d+-\d+\)', '', regex=True)  # Remove records
            .str.strip()
        )
        
        # Convert values
        table['Value'] = (
            table['Value']
            .astype(str)
            .str.replace('%', '')
            .apply(pd.to_numeric, errors='coerce')
        )
        
        if '%' in stat_url:
            table['Value'] /= 100
            
        table['Year'] = year
        table['Stat'] = STATS[stat_url]
        
        return table[['Year', 'Team', 'Stat', 'Value']].dropna()
        
    except Exception as e:
        print(f"Error scraping {stat_url} for {year}: {str(e)}")
        return pd.DataFrame()
