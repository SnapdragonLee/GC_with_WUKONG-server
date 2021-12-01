import pandas as pd

garbageListCsvPath = './garbage.csv'

typeMean = {
    1: '(可回收垃圾)',
    2: '(有害垃圾)',
    4: '(湿垃圾)',
    8: '(干垃圾)',
    16: ' (大件垃圾)'
}


class GarbageList:
    def __init__(self):
        self.dataframe = pd.read_csv(garbageListCsvPath)
        names = [name for name in self.dataframe['name']]
        types = [gType for gType in self.dataframe['type']]
        self.dic = dict(zip(names, types))

    def check(self, target: str):
        return self.dic.get(target)


def main():
    list = GarbageList()
    while True:
        s = input('您想问的垃圾的名字: ')
        s = s.split()[0]
        ans = list.check(s)
        if ans is None:
            print('库中不含此垃圾')
            continue
        garbage_types = []
        for key in typeMean.keys():
            if ans & key != 0:
                garbage_types.append(typeMean[key])
        print('垃圾种类: ' + ','.join(garbage_types))


if __name__ == '__main__':
    main()
