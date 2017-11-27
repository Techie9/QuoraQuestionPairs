# views.py

from django.shortcuts import render, HttpResponse
from kptQuoraProject.ConnectionFrontend import FrontEnd
# Create your views here.

def quora(request):
    duplicateQue = []
    frontend=FrontEnd()
    if request.method == 'POST':
        que = request.POST.get('question')
        # #duplicateQue.append(que)
        # duplicateQue.append('What are some ways to lose weight fast?')
        # duplicateQue.append('How can I slowly lose weight?')
        # duplicateQue.append('What are some ways to lose weight fast?')
        # duplicateQue.append('How can I lose 25 kg')
        # duplicateQue.append('What is the best plan to lose weight?')
        # duplicateQue.append('How can I lose weight loss?')
        # duplicateQue.append('What are some ways to lose weight fast?')
        # duplicateQue.append('What would be a realistic plan to lose weight?')
        print("question from Client :",que)
        duplicateQue=frontend.run(que)
    return render(request,'quora.html',{'data': duplicateQue})


