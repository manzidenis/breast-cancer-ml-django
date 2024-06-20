from django.shortcuts import render
import joblib
# Create your views here.

def demHome(request):
    if request.method == 'POST':
        model = joblib.load('./cancer-recommender.joblib')
    

        radius_se = float(request.POST['radius'])
        texture_se = float(request.POST['texture'])
        perimeter_se = float(request.POST['perimeter'])
        area_se = float(request.POST['area']) 
        smoothness_se = float(request.POST['smoothness']) 
        compactness_se = float(request.POST['compactness'])
        concavity_se = float(request.POST['concavity'])
        concave_points_se = float(request.POST['concavepoints'])
        symmetry_se = float(request.POST['symmetry']) 
        fractal_dimension_se = float(request.POST['fractaldimension']) 
        radius_worst = float(request.POST['radiusworst']) 
        texture_worst = float(request.POST['textureworst'])
        perimeter_worst = float(request.POST['perimeterworst'])
        area_worst = float(request.POST['areaworst'])
        smoothness_worst = float(request.POST['smoothnessworst']) 
        compactness_worst = float(request.POST['compactnessworst'])
        concavity_worst = float(request.POST['concavityworst']) 
        concave_points_worst = float(request.POST['concavepointsworst'])
        symmetry_worst = float(request.POST['symmetryworst'])
        fractal_dimension_worst = float(request.POST['fractaldimensionworst'])







                # Make predictions using the model
        
        pred = model.predict([[radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
        symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,
        concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
        check = pred[0]
        return render(request, "home.html", {'result': check})
    return render(request, "home.html" )
   
def demKing(request):
    return render(request, "home.html" )

