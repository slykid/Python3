class Store:
    def __init__(self, pizza, pasta):
        self.pizza = pizza
        self.pasta = pasta

    def total_order(self):
        return (self.pizza + self.pasta)

    @classmethod
    def same4each(cls, double):
        return cls(double, double)

    @staticmethod
    def name_of_store():
        print("이태리식당")

order1 = Store(3, 4)
print("order1 : " + str(order1.total_order()))

order2 = Store.same4each(3)
print("order2 : " + str(order2.total_order()))

order3 = Store(3, 2)
order3.name_of_store()