package com.java.kilhyun.OOP;

public class Books {

    private String name;
    private String author;

    public Books(String bookName, String authorName)
    {
        this.name = bookName;
        this.author = authorName;
    }

    public String getName()
    {
        return name;
    }

    public String getAuthorName()
    {
        return author;
    }

}
