from django.shortcuts import render
# Create your views here.




def inicio(request):
    return render(request, 'Eigen/home.html')

def reconocedor(request):
    return render(request, 'Eigen/reconocedor.html')

def trainer(request):
    return render(request, 'Eigen/trainer.html')
