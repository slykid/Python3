package com.java.kilhyun.OOP;

public class ex09_4_ObjectArrayCopy {

    public static void main(String[] args)
    {
        Book[] library = new Book[5];
        Book[] copyLib = new Book[5];


        // 반드시 assign 을 해줘야 한다.
        library[0] = new Book("Harry Potter1", "J.K Rolling");
        library[1] = new Book("Harry Potter2", "J.K Rolling");
        library[2] = new Book("Harry Potter3", "J.K Rolling");
        library[3] = new Book("Harry Potter4", "J.K Rolling");
        library[4] = new Book("Harry Potter5", "J.K Rolling");

        System.arraycopy(library, 0, copyLib, 0, 5);

        for (int i = 0; i < copyLib.length; i++)
        {
            System.out.println(library[i]);
        }

        System.out.println();

        for (int i = 0; i < copyLib.length; i++)
        {
            System.out.println(copyLib[i]);
        }

        System.out.println();

        // 향상된 for 문
        for (Book book : copyLib)
        {
            book.showBookInfo();
        }

        System.out.println();

        // 얖은 복사
        // - 변경은 library 의 객체에서 했지만 copyLib 가 library 배열의 요소인 주소를 복사해갔기 때문에
        //   library 의 변경 내용을 그대로 반영하게 된다!
        // = 주소만 복사
        library[0].setAuthor("조정래");
        library[0].setTitle("태백산맥");

        for (Book book : copyLib)
        {
            book.showBookInfo();
        }

        System.out.println();

        // 깊은 복사
        // - 객체의 값을 직접 다른 객체에 입력하는 방식
        for(int i = 0; i < copyLib.length; i++)
        {
            if(i == 0)
            {
                copyLib[0] = new Book("Sherlock", "Auther Conan Doyle");
            }
            else
            {
                copyLib[i] = new Book(library[i].getTitle(), library[i].getAuthor());
            }
        }

        for (Book book : copyLib)
        {
            book.showBookInfo();
        }

    }

}
