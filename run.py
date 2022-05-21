from src.server import Server


def run():
    svr = Server()
    while True:
        origin_sent = input("> ")
        query = [[origin_sent, ["JJ", "VB"]]]
        results = svr.inference(query)
        for i, (score, res_sent) in enumerate(results[0][origin_sent]):
            print("  {}. ({:.3f}) {}".format(str(i), score, res_sent))


if __name__ == '__main__':
    run()
