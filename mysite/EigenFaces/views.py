from django.shortcuts import render
import Trozos




def inicio(request):
    return render(request, 'Eigen/home.html')

def reconocedor(request):
    return render(request, 'Eigen/reconocedor.html')
def reconocedorProc(request):
    files = request.FILES.getlist("file")
    for i in files:
        img = i.read()
        print(Trozos.reconocer(img))
    return render(request, 'Eigen/reconocedor.html')

def trainer(request):
    return render(request, 'Eigen/trainer.html')
def trainerProc(request):
    files = request.FILES.getlist("file")
    
    porcentaje = int(request.POST['porcentaje'])
    if(files and porcentaje):
            Trozos.cargarImagen(files,porcentaje)
    return render(request, 'Eigen/trainer.html')
