from django.shortcuts import render
#from django.http import HttpResponse
from DjangoApp.models import Topic,Webpage,AccessRecords
# Create your views here.

def index(request):
    #return HttpResponse(" I love Python Programming ")
    #my_dict={'insert_me':"Hello, I am from views.py"}
    
    webpages_list=AccessRecords.objects.order_by('date')
    date_dict= {'access_records': webpages_list}
    #return render(request,"index.html",context=my_dict)
    return render(request,"index.html",context=date_dict)
    

