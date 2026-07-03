import openpyxl


class WRITE_EXCEL():
    def __init__(self, filepath):
        # open excel to write(add version) not change the original file
        self.filename = filepath
        self.wb = openpyxl.load_workbook(self.filename)
        self.ws = self.wb.active


    def run(self, results_list, line_offset, col_offset):
        for i, results in enumerate(results_list):
            self.write(1 + line_offset[i], 1 + col_offset, results['class fault'])
            self.write(2 + line_offset[i], 1 + col_offset, results['location fault'])
            self.write(3 + line_offset[i], 1 + col_offset, results['redundancy fault'])
            self.write(4 + line_offset[i], 1 + col_offset, results['missing fault'])
            self.write(5 + line_offset[i], 1 + col_offset, results['any fault'])

        # end of write

        self.save()
        self.close()

    def write(self, row, col, value):
        self.ws.cell(row=row, column=col).value = value

    def save(self):
        self.wb.save(self.filename)

    def close(self):
        self.wb.close()
