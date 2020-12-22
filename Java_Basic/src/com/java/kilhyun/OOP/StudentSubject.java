package com.java.kilhyun.OOP;

import java.util.ArrayList;

public class StudentSubject {

    public int studentId;
    public String studentName;

    ArrayList<Subject> studentSubject;


    public StudentSubject(int studentId, String studentName)
    {
        this.studentId = studentId;
        this.studentName = studentName;

        studentSubject = new ArrayList<Subject>();
    }

    public void addScore(String subjectName, int subjectScore)
    {
        Subject subject = new Subject(subjectName, subjectScore);

        studentSubject.add(subject);
    }

    public void showStudentInfo()
    {
        int sum = 0;

        for(Subject s : studentSubject)
        {
            System.out.println("학생 " + studentName + " 님의 " + s.getName() + "과목의 점수는 " + s.getScore() + "점 입니다.");
            sum += s.getScore();
        }

        System.out.println("학생 " + studentName + " 님의 총점은 " + sum +"점 입니다.");
    }

}
