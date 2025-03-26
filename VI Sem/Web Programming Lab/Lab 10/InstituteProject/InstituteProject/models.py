from django.db import models

class Institute(models.Model):
    institute_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200)
    no_of_courses = models.IntegerField()

    def __str__(self):
        return self.name
