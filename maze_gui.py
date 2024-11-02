import random
from typing import List, NamedTuple, TypeVar, Set, Dict, Optional, Callable  # Thêm import Callable
from enum import Enum
from data_structures import Node, PriorityQueue, node_to_path  # Sửa lại import statement
from tkinter import *
from tkinter.ttk import *

T = TypeVar('T')

# Định nghĩa các loại ô trong mê cung
class Cell(str, Enum):
    EMPTY = " "
    BLOCKED = "X"
    START = "S"
    GOAL = "G"
    EXPLORED = "E"
    CURRENT = "C"
    FRONTIER = "F"
    PATH = "*"

# Định nghĩa vị trí trong mê cung
class MazeLocation(NamedTuple):
    row: int
    column: int  # Thêm kiểu dữ liệu cho column

    def __str__(self):
        return f"({self.row}, {self.column})"

# Lớp GUI cho mê cung
class MazeGUI:
    def __init__(self, rows: int = 20, columns: int = 20, sparseness: float = 0.2,
                 start: MazeLocation = MazeLocation(0, 0), goal: MazeLocation = MazeLocation(9, 9)) -> None:
        # Khởi tạo các biến cơ bản
        self._rows: int = rows
        self._columns: int = columns
        self.start: MazeLocation = start
        self.goal: MazeLocation = goal
        # Điền các ô trống vào lưới
        self._grid: List[List[Cell]] = [[Cell.EMPTY for c in range(columns)] for r in range(rows)]
        # Điền các ô bị chặn vào lưới
        self._randomly_fill(rows, columns, sparseness)
        # Điền vị trí bắt đầu và kết thúc
        self._grid[start.row][start.column] = Cell.START
        self._grid[goal.row][goal.column] = Cell.GOAL
        self._setup_GUI()
        self._setting_goal = False # Biến để kiểm tra xem người dùng có đang chọn mục tiêu không
        self.root.mainloop()  # Đảm bảo mainloop được gọi ở đây

    # Thiết lập GUI
    def _setup_GUI(self):
        # Bắt đầu GUI
        self.root: Tk = Tk()
        self.root.title("Maze Solving")
        Grid.rowconfigure(self.root, 0, weight=1)
        Grid.columnconfigure(self.root, 0, weight=1)
        # Cửa sổ chính
        frame: Frame = Frame(self.root)
        frame.grid(row=0, column=0, sticky=N + S + E + W)
        # Thiết lập style cho các widget
        style: Style = Style()
        style.theme_use('classic')
        style.configure("BG.TLabel", foreground="black", font=('Helvetica', 14))  
        style.configure("BG.TButton", foreground="black", font=('Helvetica', 14))  
        style.configure("BG.TListbox", foreground="black", font=('Helvetica', 14))  
        style.configure("BG.TCombobox", foreground="black", font=('Helvetica', 14))  
        style.configure(" ", foreground="black", background="white")
        style.configure(Cell.EMPTY.value + ".TLabel", foreground="black", background="white", font=('Helvetica', 14))  
        style.configure(Cell.BLOCKED.value + ".TLabel", foreground="white", background="black", font=('Helvetica', 14))  
        style.configure(Cell.START.value + ".TLabel", foreground="black", background="green", font=('Helvetica', 14)) 
        style.configure(Cell.GOAL.value + ".TLabel", foreground="black", background="red", font=('Helvetica', 14)) 
        style.configure(Cell.PATH.value + ".TLabel", foreground="black", background="cyan", font=('Helvetica', 14))  
        style.configure(Cell.EXPLORED.value + ".TLabel", foreground="black", background="yellow", font=('Helvetica', 14)) 
        style.configure(Cell.CURRENT.value + ".TLabel", foreground="black", background="blue", font=('Helvetica', 14))  
        style.configure(Cell.FRONTIER.value + ".TLabel", foreground="black", background="orange", font=('Helvetica', 14)) 
        # Đặt các nhãn ở bên cạnh
        for row in range(self._rows):
            Grid.rowconfigure(frame, row, weight=1)
            row_label: Label = Label(frame, text=str(row), style="BG.TLabel", anchor="center")
            row_label.grid(row=row, column=0, sticky=N + S + E + W)
            Grid.rowconfigure(frame, row, weight=1)
            Grid.grid_columnconfigure(frame, 0, weight=1)
        # Đặt các nhãn ở dưới cùng
        for column in range(self._columns):
            Grid.columnconfigure(frame, column + 1, weight=1)  # Đảm bảo tất cả các cột có trọng số bằng nhau
            column_label: Label = Label(frame, text=str(column), style="BG.TLabel", anchor="center")
            column_label.grid(row=self._rows, column=column + 1, sticky=N + S + E + W)
            Grid.rowconfigure(frame, self._rows, weight=1)
            Grid.columnconfigure(frame, column + 1, weight=1)
        # Thiết lập hiển thị lưới
        self._cell_labels: List[List[Label]] = [[Label(frame, anchor="center", borderwidth=1, relief="solid") for c in range(self._columns)] for r in
                                                range(self._rows)]  # Thêm borderwidth và relief để tạo lưới
        for row in range(self._rows):
            for column in range(self._columns):
                cell_label: Label = self._cell_labels[row][column]
                cell_label.grid(row=row, column=column + 1, sticky=N + S + E + W, padx=0, pady=0)  # Giảm padding
                Grid.columnconfigure(frame, column + 1, weight=1)  # Đảm bảo tất cả các cột có trọng số bằng nhau
        self._display_grid()
        # Thiết lập các nút
        # Nút chạy thuật toán
        astar_button: Button = Button(frame, style="BG.TButton", text="Run Astar Algorithm", command=self.run_astar)
        astar_button.grid(row=self._rows + 2, column=0, columnspan=6, sticky=N + S + E + W)
        Grid.rowconfigure(frame, self._rows + 2, weight=1)
        #nút randomize mê cung
        randomize_button: Button = Button(frame, style="BG.TButton", text="Randomize Maze", command=self.randomize_maze)
        randomize_button.grid(row=self._rows + 3, column=0, columnspan=6, sticky=N + S + E + W)
        Grid.rowconfigure(frame, self._rows + 3, weight=1)
        # Nút đặt mục tiêu
        set_goal_button: Button = Button(frame, style="BG.TButton", text="Set Goal", command=self.set_goal)
        set_goal_button.grid(row=self._rows + 4, column=0, columnspan=6, sticky=N + S + E + W)
        Grid.rowconfigure(frame, self._rows + 4, weight=1)
        # Thiết lập hiển thị cấu trúc dữ liệu
        frontier_label: Label = Label(frame, text="Frontier", style="BG.TLabel", anchor="center")
        frontier_label.grid(row=0, column=self._columns + 2, columnspan=3, sticky=N + S + E + W)
        explored_label: Label = Label(frame, text="Explored", style="BG.TLabel", anchor="center")
        explored_label.grid(row=self._rows // 2, column=self._columns + 2, columnspan=3, sticky=N + S + E + W)
        Grid.columnconfigure(frame, self._columns + 2, weight=1)
        Grid.columnconfigure(frame, self._columns + 3, weight=1)
        Grid.columnconfigure(frame, self._columns + 4, weight=1)
        self._frontier_listbox: Listbox = Listbox(frame, font=("Helvetica", 14))
        self._frontier_listbox.grid(row=1, column=self._columns + 2, columnspan=3, rowspan=self._rows // 2 - 1,
                                    sticky=N + S + E + W, padx=0, pady=0)  # Giảm padding
        self._explored_listbox: Listbox = Listbox(frame, font=("Helvetica", 14))
        self._explored_listbox.grid(row=self._rows // 2 + 1, column=self._columns + 2, columnspan=3,
                                    rowspan=self._rows // 2 - 1, sticky=N + S + E + W, padx=0, pady=0)  # Giảm padding
        # Spinbox cho khoảng thời gian
        interval_label: Label = Label(frame, text="Interval", style="BG.TLabel", anchor="center")
        interval_label.grid(row=self._rows + 1, column=self._columns + 2, columnspan=3, sticky=N + S + E + W)
        self._interval_box: Combobox = Combobox(frame, state="readonly", values=[0.1, 0.2, 0.5, 1, 2], justify="center",
                                                style="BG.TCombobox")
        self._interval_box.set(0.2)  # Set default to faster speed
        self._interval_box.grid(row=self._rows + 2, column=self._columns + 2, columnspan=3, sticky=N + S + E + W, padx=0, pady=0)  # Giảm padding
        # Đóng gói và hiển thị
        frame.pack(fill="both", expand=True)

    # Điền ngẫu nhiên các ô bị chặn vào lưới
    def _randomly_fill(self, rows: int, columns: int, sparseness: float):
        for row in range(rows):
            for column in range(columns):
                if random.uniform(0, 1.0) < sparseness:
                    self._grid[row][column] = Cell.BLOCKED

    # Hiển thị lưới
    def _display_grid(self):
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
        for row in range(self._rows):
            for column in range(self._columns):
                cell: Cell = self._grid[row][column]
                cell_label: Label = self._cell_labels[row][column]
                cell_label.configure(style=cell.value + ".TLabel")
                cell_label.bind("<Button-1>", lambda e, r=row, c=column: self.on_cell_click(r, c))  # Thêm click chuột
    #chọn Goal
    def on_cell_click(self, row: int, column: int):
        if self._setting_goal:
            if self._grid[row][column] != Cell.BLOCKED:     
                self._grid[self.goal.row][self.goal.column] = Cell.EMPTY  
                self.goal = MazeLocation(row, column)
                self._grid[row][column] = Cell.GOAL
                self._setting_goal = False
                self._display_grid()

    def set_goal(self):
        self._setting_goal = True
        print("Click on a cell to set it as the goal.")  #

    # Thực hiện một bước trong thuật toán
    def step(self, frontier, explored, costs, last_node):
        if isinstance(frontier, PriorityQueue) and not frontier.empty: 
            current_node: Node[T] = frontier.pop()
            current_state: T = current_node.state
            self._frontier_listbox.delete(0, 0)
            self._grid[current_state.row][current_state.column] = Cell.CURRENT
            if last_node is not None:
                self._grid[last_node.state.row][last_node.state.column] = Cell.EXPLORED
            # nếu tìm thấy đích, kết thúc
            if self.goal_test(current_state):
                path = node_to_path(current_node)
                self.mark(path)
                self._display_grid()
                return 
           # Kiểm tra các vị trí kế tiếp
            for child in self.successors(current_state):
                new_cost = current_node.cost + 1
                if child not in costs or new_cost < costs[child]:
                    costs[child] = new_cost
                    priority = new_cost + self.euclidean_distance(self.goal)(child)
                    frontier.push(Node(child, current_node, new_cost, priority))
                    explored.add(child)
                    # Update GUI
                    self._grid[child.row][child.column] = Cell.FRONTIER
                    self._explored_listbox.insert(END, str(child))
                    self._frontier_listbox.insert(END, str(child))
                    self._explored_listbox.select_set(END)
                    self._explored_listbox.yview(END)
                    self._frontier_listbox.select_set(END)
                    self._frontier_listbox.yview(END)
            self._display_grid()
            num_delay = int(float(self._interval_box.get()) * 1000)  # Convert to milliseconds
            self.root.after(num_delay, self.step, frontier, explored, costs, current_node)
        else:
            # Nếu không còn phần tử nào trong hàng đợi
            print("méo thấy.")

    # Chạy thuật toán A*
    def run_astar(self):
        self.clear()
        initial_node = Node(self.start, None, 0.0, self.euclidean_distance(self.goal)(self.start)) #hoạc dùng hàm manhattan_distance
        frontier = PriorityQueue()
        frontier.push(initial_node)
        explored = set()
        costs = {self.start: 0.0}
        self.step(frontier, explored, costs, None)

    def astar(self, initial: MazeLocation, goal_test: Callable[[MazeLocation], bool], successors: Callable[[MazeLocation], List[MazeLocation]], heuristic: Callable[[MazeLocation], float]) -> Optional[Node[MazeLocation]]:
        frontier = PriorityQueue[Node[MazeLocation]]()
        start_node = Node(state=initial, parent=None, cost=0.0, heuristic=heuristic(initial))
        frontier.push(start_node)
        explored: Dict[MazeLocation, float] = {initial: 0.0}

        while not frontier.empty:
            current_node = frontier.pop()
            current_state = current_node.state

            if goal_test(current_state):
                return current_node

            for child in successors(current_state):
                new_cost = current_node.cost + 1  # Giả sử chi phí di chuyển giữa các ô là 1
                if child not in explored or explored[child] > new_cost:
                    explored[child] = new_cost
                    child_node = Node(state=child, parent=current_node, cost=new_cost, heuristic=heuristic(child))
                    frontier.push(child_node)

        return None

    def goal_test(self, ml: MazeLocation) -> bool:
        return ml == self.goal

    def heuristic(self, ml: MazeLocation) -> float:
        return MazeGUI.euclidean_distance(self.goal)(ml)

    def find_path(self) -> List[MazeLocation]:
        solution = self.astar(self.start, self.goal_test, self.successors, self.heuristic)
        if solution is None:
            return []
        else:
            return node_to_path(solution)

    # Tìm các vị trí kế tiếp có thể đi tới
    def successors(self, ml: MazeLocation) -> List[MazeLocation]:
        locations: List[MazeLocation] = []
        if ml.row + 1 < self._rows and self._grid[ml.row + 1][ml.column] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row + 1, ml.column))
        if ml.row - 1 >= 0 and self._grid[ml.row - 1][ml.column] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row - 1, ml.column))
        if ml.column + 1 < self._columns and self._grid[ml.row][ml.column + 1] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row, ml.column + 1))
        if ml.column - 1 >= 0 and self._grid[ml.row][ml.column - 1] != Cell.BLOCKED:
            locations.append(MazeLocation(ml.row, ml.column - 1))
        return locations

    # Đánh dấu đường đi trên lưới
    def mark(self, path: List[MazeLocation]):
        for maze_location in path:
            self._grid[maze_location.row][maze_location.column] = Cell.PATH
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
    
    # Tính khoảng cách Euclidean
    @staticmethod
    def euclidean_distance(goal: MazeLocation) -> Callable[[MazeLocation], float]:  
        def distance(ml: MazeLocation) -> float:
            xdist: int = ml.column - goal.column
            ydist: int = ml.row - goal.row
            return (xdist ** 2 + ydist ** 2) ** 0.5
        return distance
    # Xóa các dấu vết trên lưới
    def clear(self):
        self._frontier_listbox.delete(0, END)
        self._explored_listbox.delete(0, END)
        for row in range(self._rows):
            for column in range(self._columns):
                if self._grid[row][column] != Cell.BLOCKED:
                    self._grid[row][column] = Cell.EMPTY
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL

    # Thay đổi mê cung
    def randomize_maze(self):
        self._grid = [[Cell.EMPTY for c in range(self._columns)] for r in range(self._rows)]  # Đặt lại tất cả các ô thành ô trống
        self._randomly_fill(self._rows, self._columns, 0.2)
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
        self._display_grid()

if __name__ == "__main__":
    m: MazeGUI = MazeGUI()
