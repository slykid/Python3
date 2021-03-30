def unicode_test(value):
    import unicodedata

    name = unicodedata.name(value)
    value2 = unicodedata.lookup(name)

    print("value {}, name={}, value2={}".format(value, name, value2))

unicode_test('A')
unicode_test('$')
unicode_test('\u20ac')