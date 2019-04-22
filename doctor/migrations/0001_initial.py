# Generated by Django 2.1.7 on 2019-03-28 16:04

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('name', models.CharField(max_length=255)),
                ('phone', models.CharField(max_length=18)),
                ('email', models.CharField(max_length=255)),
                ('location', models.CharField(max_length=500)),
                ('id', models.CharField(max_length=20, primary_key=True, serialize=False)),
                ('password', models.CharField(max_length=255)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete='cascade', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]