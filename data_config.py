
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = '/home/cver2080/4TDISK/wjd/datasets/LEVIR-CD256/'  # Your Path
        elif data_name == 'BCD':
            self.root_dir = '/home/cver2080/4TDISK/wjd/datasets/BCD/'
        elif data_name == 'GZCD':
            self.root_dir = '/home/cver2080/4TDISK/wjd/datasets/GZ-CD256/'    
        elif data_name == 'EGYBCD':
            self.root_dir = '/home/cver2080/4TDISK/wjd/datasets/EGY-BCD/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    # print(data.data_name)
    # print(data.root_dir)
    # print(data.label_transform)

