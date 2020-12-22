package com.java.kilhyun.OOP;

import java.util.ArrayList;

public class StudentRecord {

    public int studentId;
    public String studentName;

    ArrayList<Books> record;

    public StudentRecord(int id, String name)
    {
        this.studentId = id;
        this.studentName = name;

        record = new ArrayList<Books>();
    }

    public void addRecord(String bookName, String authorName)
    {
        Books books = new Books(bookName, authorName);

        record.add(books);
    }

    public void showBookRecord()
    {
        System.out.print(studentName + " 학생이 읽은 책은 : ");

        int iter = 0;
        for(Books books : record)
        {
            System.out.print(books.getName());
            iter += 1;

            if (iter < record.size()) {
                System.out.print(", ");
            }
        }

        System.out.println(" 입니다.");

    }

}
