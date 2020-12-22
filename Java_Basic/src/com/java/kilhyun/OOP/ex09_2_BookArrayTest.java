package com.java.kilhyun.OOP;

public class ex09_2_BookArrayTest {

    public static void main(String[] args)
    {
        Book[] library = new Book[5];

        // 반드시 assign 을 해줘야 한다.
        library[0] = new Book("Harry Potter1", "J.K Rolling");
        library[1] = new Book("Harry Potter2", "J.K Rolling");
        library[2] = new Book("Harry Potter3", "J.K Rolling");
        library[3] = new Book("Harry Potter4", "J.K Rolling");
        library[4] = new Book("Harry Potter5", "J.K Rolling");

        for (int i = 0; i < library.length; i++)
        {
            System.out.println(library[i]);
        }

    }

}
