import turtle
loadWindow = turtle.Screen()
turtle.speed(1)
turtle.colormode(255)
turtle.pensize(1)

STEP = 25
LENGTH = 1000

if __name__ == '__main__':
    for i in range(0, LENGTH, STEP):
       turtle.penup()
       turtle.setpos(-LENGTH/2, LENGTH/2 - i)
       turtle.pendown()
       turtle.setpos(LENGTH/2, LENGTH/2 - i)
       turtle.penup()
       turtle.setpos(-LENGTH/2 + i, LENGTH/2)
       turtle.pendown()
       turtle.setpos(-LENGTH/2 + i, -LENGTH/2)