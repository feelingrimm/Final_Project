app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'
# Loading the model and scaler
cancer_model = load_model("cancer_data_model.h5")
cancer_scaler = joblib.load("cancer_data.pkl")
# Now create a WTForm Class
class CancerForm(FlaskForm):
 text_mean = TextField('Cell Texture Mean')
 per_mean = TextField('Cell Perimeter Mean')
 smo_mean = TextField('Cell Smoothness Mean')
 sym_mean = TextField('Cell Symmetry Mean')
 frac_mean = TextField('Cell Fractal Dimension Mean')
 text_se = TextField('Cell Radius Standard Error')
 smo_se = TextField('Cell Smoothness Standard Error')
 comp_se = TextField('Cell Compactness Standard Error')
 sym_se = TextField('Cell Symmetry Standard Error')
 
@app.route('/', methods=['GET', 'POST'])
def index():
  # Create instance of the form.
   form = CancerForm()
  # If the form is valid on submission
  if form.validate_on_submit():
  # Grab the data from the input on the form.
  session['text_mean'] = form.text_mean.df
  session['per_mean'] = form.per_mean.df
  session['smo_mean'] = form.smo_mean.df
  session['sym_mean'] = form.sym_mean.df
  session['frac_mean'] = form.frac_mean.df
  session['text_se'] = form.text_se.df
  session['smo_se'] = form.smo_se.df
  session['comp_se'] = form.comp_se.df
  session['sym_se'] = form.sym_se.df
  
return redirect(url_for(“prediction”))
return render_template('home.html', form=form)
@app.route('/prediction')
def prediction():
 #Defining content dictionary
 content = {}
 content['texture_mean'] = float(session['text_mean'])
 content['perimeter_mean'] = float(session['per_mean'])
 content['smoothness_mean'] = float(session['smo_mean'])
 content['symmetry_mean'] = float(session['sym_mean'])
 content['fractal_dimension_mean'] = float(session['frac_mean'])
 content['texture_se'] = float(session['text_se'])
 content['smoothness_se'] = float(session['smo_se'])
 content['compactness_se'] = float(session['comp_se'])
 content['symmetry_se'] = float(session['sym_se'])

 
 results = return_prediction(model=cancer_model,scaler=cancer_scaler,sample_json=content)
return render_template('prediction.html',results=results)
if __name__ == '__main__':
 app.run(debug=True)