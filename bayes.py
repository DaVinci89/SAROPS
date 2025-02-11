import sys
import random
import itertools
import numpy as np
import cv2 as cv


"""Симуляція роботи програми SAROPS(оптимальне планування пошуково-рятувальних операцій. Пошук пропавшої людини в 
трьох суміжних областях. Відображення карти, вивід меню пошуку, довільний вибір місцерозташування людини і 
відображення її в разі успішного знаходження, або виконання байесовського алгоритму для імовірностей знаходження в 
кожній області."""

# Карта для пошукових операцій
MAP_FILE = "img/search_map.png"

# Константи для областей пошуку(прямокутники з координатами в пікселях, верхній лівий та нижній правий x i y)
SA1_CORNERS = (130, 265, 180, 315)
SA2_CORNERS = (80, 255, 130, 305)
SA3_CORNERS = (105, 205, 155, 255)


class Search:
    """Шаблон для створення пошукової місії в 3 областях"""

    def __init__(self, name):
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print(f"Неможливо завантажити зображення карти {MAP_FILE}")
            sys.exit(1)
        self.area_actual = 0  # Кількість областей пошуку
        self.human_actual = [0, 0]  # Координати місцезнаходження людини

        # Підмасиви для роботи з локальними координатами в кожній області пошуку(верхній лівий - нижній правий y i x)
        self.sa1 = self.img[SA1_CORNERS[1]:SA1_CORNERS[3],
                   SA1_CORNERS[0]:SA1_CORNERS[2]]
        self.sa2 = self.img[SA2_CORNERS[1]:SA2_CORNERS[3],
                   SA2_CORNERS[0]:SA2_CORNERS[2]]
        self.sa3 = self.img[SA2_CORNERS[1]:SA2_CORNERS[3],
                   SA2_CORNERS[0]:SA2_CORNERS[2]]

        # Попередні ймовірності місцезнаходження людини для 3 областей
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        # Показники ефективності пошуку
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    def draw_map(self, last_known):
        """Відображення карти з масштабом, останніми відомими координатами xy, і областями пошуку"""

        # Масштабний відрізок(аргументи - карта, кортеж лівих і правих координат ху, кортеж кольору відрізка,
        # ширина відрізка)
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)

        # Підпис для масштабного відрізка
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))
        cv.putText(self.img, '50 миль', (71, 370), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))

        # Створення областей пошуку згідно координат і підписи до них
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_COMPLEX, 0.6, 0)
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_COMPLEX, 0.6, 0)
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_COMPLEX, 0.6, 0)

        # Мітка + на останньому відомому місцезнаходженні
        cv.putText(self.img, '+', last_known, cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
        cv.putText(self.img, '+ = Последнее известное местоположение', (225, 355), cv.FONT_HERSHEY_COMPLEX, 0.35,
                   (0, 135, 255))
        cv.putText(self.img, '* = Текущее местоположение', (226, 370), cv.FONT_HERSHEY_COMPLEX, 0.35, (255, 135, 0))

        cv.imshow('SEARCH AREA', self.img)
        cv.moveWindow('SEARCH AREA', 10, 10)
        cv.waitKey(500)

    def human_final_location(self, num_search_areas):
        """Повертає координати х,у зниклої людини"""
        # Пошук координат людини в відношенні будь-якого підмасива області пошуку
        self.human_actual[0] = np.random.choice(self.sa1.shape[1])
        self.human_actual[1] = np.random.choice(self.sa1.shape[0])

        area = int(random.triangular(1, num_search_areas + 1))
        if area == 1:
            x = self.human_actual[0] + SA1_CORNERS[0]
            y = self.human_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.human_actual[0] + SA2_CORNERS[0]
            y = self.human_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        if area == 3:
            x = self.human_actual[0] + SA3_CORNERS[0]
            y = self.human_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        return x, y

    def calc_search_effectiveness(self):
        """Метод для випадкового визначення ефективності пошуку"""
        # Установка десятичних значень ефективності пошуку для кожної області пошуку
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob):
        """Результати пошуку і список побачених координат"""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(itertools.product(local_x_range, local_y_range))
        random.shuffle(coords)
        coords = coords[:int((len(coords) * effectiveness_prob))]
        loc_actual = (self.human_actual[0], self.human_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f"Знайдено в Зоні {area_num}", coords
        else:
            return "Не знайдено", coords

    def revise_target_probs(self):
        """Застосування теореми Байеса. Оновлення ймовірності цілей в області на основі ефективності пошуку"""
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)
        self.p1 = self.p1 * (1 - self.sep1) / denom
        self.p2 = self.p2 * (1 - self.sep2) / denom
        self.p3 = self.p3 * (1 - self.sep3) / denom

def draw_menu(search_num):
    """Створення меню вибору для пошуку в області"""
    print("\n\t\tSAROPS MENU\n\n")
    print(f"ПОШУК {search_num}")
    print(
        """
        Виберіть наступні зони для пошуку:
        
        0 >> Вихід
        1 >> Пошук в Зоні 1 двічі
        2 >> Пошук в Зоні 2 двічі
        3 >> Пошук в Зоні 3 двічі
        4 >> Пошук в Зонах 1 і 2
        5 >> Пошук в Зонах 1 і 3
        6 >> Пошук в Зонах 2 і 3
        7 >> Рестарт
        """
    )

def main():
    app = Search("HUMAN SEARCH")
    app.draw_map(last_known=(160, 290))
    human_x, human_y = app.human_final_location(num_search_areas=3)
    print("-" * 65)
    print("\nІніціалізація цільових імовірностей(P):")
    print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))
    search_num = 1
    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input("Зробіть ваш вибір: ")
        if choice == "0":
            sys.exit()
        elif choice == "1":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            app.sep1 = (len(set(coords_1 + coords_2))) / (len(app.sa1)**2)
            app.sep2 = 0
            app.sep3 = 0
        elif choice == "2":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2))) / (len(app.sa2)**2)
            app.sep3 = 0
        elif choice == "3":
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2))) / (len(app.sa3)**2)
        elif choice == "4":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0
        elif choice == "5":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep2 = 0
        elif choice == "6":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
        elif choice == "7":
            main()
        else:
            print("\nВибачте, Ваш вибір некоректний", file=sys.stderr)
            continue

        app.revise_target_probs()
        print(f"\nПошук {search_num} Результат 1 {results_1}", file=sys.stderr)
        print(f"\nПошук {search_num} Результат 2 {results_2}", file=sys.stderr)
        print(f"\nПошук {search_num} Ефективність (Е):", file=sys.stderr)
        print("E1 = {:.3f}, E2 = {:.3f}, E3 = {:.3F}".format(app.sep1, app.sep2, app.sep3))
        if results_1 == "Не знайдено" and results_2 == "Не знайдено":
            print(f"\nНові цільові імовірності (P) для пошуку {search_num + 1}:")
            print("P1 = {:.3f}, P2 = {:.3f}, P3 = {:.3f}".format(app.p1, app.p2, app.p3))
        else:
            cv.circle(app.img, (human_x, human_y), 3, (255, 0, 0), -1)
            cv.imshow("SEARCH AREA", app.img)
            cv.waitKey(1500)
            main()
        search_num += 1

if __name__ == "__main__":
    main()
