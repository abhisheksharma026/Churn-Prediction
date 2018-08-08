from django.contrib import admin

# Register your models here.

from DjangoApp.models import Topic,Webpage,AccessRecords
admin.site.register(Topic)
admin.site.register(Webpage)
admin.site.register(AccessRecords)

