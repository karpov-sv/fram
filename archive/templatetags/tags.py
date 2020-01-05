from __future__ import absolute_import, division, print_function, unicode_literals

from django import template

register = template.Library()

@register.simple_tag(takes_context=True)
def relative_root(context):
    #return context['request'].path
    return ""
