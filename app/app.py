from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
# model = load('model-deploy.joblib')
model=pickle.load(open("modelp-deploy.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        founded_at = request.form['Founded_at']
        created_at = request.form['Created_at']
        updated_at = request.form['Updated_at']
        closed_at = request.form['Closed_at']
        company_age = request.form['Company_age']
        investment_rounds = request.form['Investment_rounds']
        funding_rounds = request.form['Funding_rounds']
        funding_total_usd = request.form['Funding_total_usd']
        milestones = request.form['milestones']
        relationships = request.form['relationships']
        return_on_investment = request.form['Return_on_investment']
        
        category_code = request.form['Category_code']
        consulting, advertising, biotech, ecommerce, games_video, enterprise, software, web, mobile, others = [0]*10
        if category_code == 'Consulting':
            consulting = 1
        elif category_code == 'Advertising':
            advertising = 1
        elif category_code == 'Biotech':
            biotech = 1
        elif category_code == 'Ecommerce':
            ecommerce = 1
        elif category_code == 'Games_Video':
            games_video = 1
        elif category_code == 'Enterprise':
            enterprise = 1
        elif category_code == 'Software':
            software = 1
        elif category_code == 'Web':
            web = 1
        elif category_code == 'Mobile':
            mobile = 1
        elif category_code == 'Other':
            others = 1

        country_code = request.form['Country_code']
        aus, can, esp, deu, fra, gbr, ind, isr, nld, usa, other = [0]*11
        if country_code == 'AUS':
            aus = 1
        elif country_code == 'CAN':
            can = 1
        elif country_code == 'ESP':
            esp = 1
        elif country_code == 'DEU':
            deu = 1
        elif country_code == 'FRA':
            fra = 1
        elif country_code == 'GBR':
            gbr = 1
        elif country_code == 'IND':
            ind = 1
        elif country_code == 'ISR':
            isr = 1
        elif country_code == 'NLD':
            nld = 1
        elif country_code == 'USA':
            usa = 1
        elif country_code == 'Other':
            other = 1

        y_pred = model.predict([[founded_at, closed_at, investment_rounds, funding_rounds, funding_total_usd, milestones,
                                 relationships, created_at, updated_at, return_on_investment, company_age, advertising,
                                 biotech, consulting, ecommerce, enterprise, games_video, mobile, others, software, web,
                                 aus, can, deu, esp, fra, gbr, ind, isr, nld, usa, other]])

        result = 'Operating' if y_pred[0] == 1 else 'Closed'
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
