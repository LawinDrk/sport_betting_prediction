import requests
from textblob import TextBlob
from newspaper import Article, Config

API_KEY = "6618a348eb5944a59d63a831bdc875d2"

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config.request_timeout = 8 # On réduit le timeout pour ne pas perdre de temps sur les sites lents

def get_sentiment(team_name):
    print(f"\n🔍 ANALYSE DE SENTIMENT : {team_name}")
    query = f'"{team_name}" AND "Premier League"'
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=60&apiKey={API_KEY}"
    
    try:
        r = requests.get(url).json()
        articles_data = r.get('articles', [])
        valid_scores = []
        seen_urls = set()

        for a_data in articles_data:
            link = a_data.get('url')
            title = a_data.get('title', '')

            # Evite les domaines de consentements, ou les sites qui bloquent
            if any(domain in link for domain in ["yahoo.com", "consent.", "msn.com", "google.com"]):
                continue

            if link in seen_urls or team_name.lower() not in title.lower():
                continue
            
            seen_urls.add(link)
            print(f"Trzitement : {title}")

            try:
                scraper = Article(link, config=config)
                scraper.download()
                scraper.parse()
                
                # Nettoyage du texte pour TextBlob
                text_content = scraper.text
                if not text_content:
                    print(" Article vide ou mal extrait.")
                    continue

                blob = TextBlob(text_content)
                keyword = team_name.split()[-1].lower()
                
                # Extraction de toutes les phrases
                relevant_sentences = [str(s).strip() for s in blob.sentences if keyword in str(s).lower()]
                
                if relevant_sentences:
                    print(f"{len(relevant_sentences)} phrases identifiées :")
                    for i, sent in enumerate(relevant_sentences):
                        # Affichage de la ligne
                        print(f"      [{i+1}] {sent[:150]}...") 
                    
                    score = TextBlob(" ".join(relevant_sentences)).sentiment.polarity
                    valid_scores.append(score)
                    print(f"   📈 Polarité calculée : {score:.2f}")
                else:
                    print("Aucune phrase spécifique à l'équipe trouvée dans le corps.")
                
            except Exception as e:
            
                print(f"   ❌ Échec technique sur cette source.")
                
            if len(valid_scores) >= 8: # On s'arrête à 8 articles de qualité
                break
                
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0
    except Exception as e:
        print(f"Erreur API News : {e}")
        return 0